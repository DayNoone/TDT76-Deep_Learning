import glob

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import os
import numpy as np
from helpers.helpers import print_progress
from helpers.helpers import l2norm
import pickle


LAYER = 'fc2'

def get_model():
	base_model = ResNet50(weights='imagenet', include_top=False)
	return base_model
	# return Model(input=base_model.input, output=base_model.get_layer(LAYER).output)


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
	from sklearn.metrics.pairwise import cosine_similarity
	vgg = get_model()
	objects1 = predict(vgg, "./../train/pics/000038100/9b4e2b7210e3e36f.jpg")
	print(objects1)
	# objects2 = predict(vgg, "785baa9024730774.jpg")
	# print(len(objects1[0]))
	# id, label, prob = decode_predictions(objects2, top=1)[0][0]
	# print(cosine_similarity(objects1, objects2))


def run_vgg():
	vgg = get_model()
	for folder_path in glob.glob("./../train/pics/*"):
		count = 0
		tot = len(glob.glob(folder_path + "/*.jpg"))
		predictions = {}
		if not os.path.isfile(folder_path):
			for filepath in glob.glob(folder_path + "/*.jpg"):
				if os.stat(filepath).st_size > 100:
					predictions[get_id_from_path(filepath)] = predict(vgg, filepath)[0][0]
				print_progress(count, tot, prefix=folder_path)
				count += 1
			image_embedding_file = open("stored_image_embeddings/" + folder_path.split('/')[-1] + ".pickle", 'wb')
			pickle.dump(predictions, image_embedding_file, protocol=2)
			image_embedding_file.close()
			print()
		else:
			print("Skipping folder: ", folder_path)


def check_for_corrupt_images():
	count = 0
	for folder_path in glob.glob("./../train/pics/*"):
		for filepath in glob.glob(folder_path + "/*.jpg"):
			try:
				image.load_img(filepath, target_size=(224, 224))
			except:
				print(filepath)
				count += 1
	print(count)

if __name__ == "__main__":
	run_vgg()
	# check_for_corrupt_images()