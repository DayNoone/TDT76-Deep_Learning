import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import numpy as np
from helpers.helpers import print_progress
import pickle

def get_model():
	base_model = ResNet50(weights='imagenet', include_top=False)
	return base_model
	# return Model(input=base_model.input, output=base_model.get_layer(LAYER).output)


LAYER = 'fc2'
print("Getting model...")
model = get_model()



def predict(model, img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return model.predict(x)


def get_id_from_path(file):
	image_id = (file.split(".jpg")[0]).split("/")[-1]
	return image_id

def run_test():
	# from sklearn.metrics.pairwise import cosine_similarity
	vgg = get_model()
	objects1 = predict(vgg, "./../train/pics/000038100/9b4e2b7210e3e36f.jpg")
	print(objects1)
	# objects2 = predict(vgg, "785baa9024730774.jpg")
	# print(len(objects1[0]))
	# id, label, prob = decode_predictions(objects2, top=1)[0][0]
	# print(cosine_similarity(objects1, objects2))


def run_vgg(folder, check_for_corrupt_images=False):
	if check_for_corrupt_images:
		if print_corrupt_images() > 0:
			return
	for folder_path in glob.glob(folder + "pics/*"):
		store_path = "preprocessing/stored_image_embeddings_" + folder.split("/")[1] + "/" + folder_path.split('/')[-1] + ".pickle"
		count = 1
		tot = len(glob.glob(folder_path + "/*.jpg"))
		predictions = {}
		if not os.path.isfile(store_path):
			for filepath in glob.glob(folder_path + "/*.jpg"):
				if os.stat(filepath).st_size > 100:
					predictions[get_id_from_path(filepath)] = predict(model, filepath)[0][0][0]
				print_progress(count, tot, prefix=folder_path)
				count += 1
			image_embedding_file = open(store_path, 'wb')
			pickle.dump(predictions, image_embedding_file, protocol=2)
			image_embedding_file.close()
			print()
		else:
			print("Already preprocessed folder: ", folder_path)

def embed_image(path):
	return predict(model, path)[0][0][0]


def print_corrupt_images():
	count = 0
	for folder_path in glob.glob("./../train/pics/*"):
		for filepath in glob.glob(folder_path + "/*.jpg"):
			try:
				image.load_img(filepath, target_size=(224, 224))
			except:
				print("Warning, corrupt image: ", filepath)
				count += 1
	return count


def check_similarity():
	vector1 = "f1dd0d51388c40b0"
	vector2 = "d75c52b8f629543f"
	dissimilar = "634514be57b43ada"

	f = open("preprocessing/stored_image_embeddings_train/000000000.pickle", "rb")
	image_dict1 = pickle.load(f)
	f.close()

	from sklearn.metrics.pairwise import cosine_similarity
	vec1 = image_dict1[vector1]
	vec2 = image_dict1[vector2]
	sim1 = cosine_similarity(vec1, vec2)[0][0]
	print("Similar: ", sim1)

	# vec1 = image_dict1[vector1]
	# vec2 = image_dict1[dissimilar]
	# sim2 = cosine_similarity(vec1, vec2)[0][0]
	# print("Dissimilar", sim2)




if __name__ == "__main__":
	# run_vgg("train")
	check_similarity()