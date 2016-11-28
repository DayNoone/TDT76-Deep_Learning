import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import os
import numpy as np
from helpers import print_progress
import pickle


def get_model():
	base_model = ResNet50(weights='imagenet', include_top=False)
	return base_model


LAYER = 'fc2'
print("Loading image embedding model...")
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


def run_vgg(folder, check_for_corrupt_images=False):
	if check_for_corrupt_images:
		if print_corrupt_images() > 0:
			return
	for folder_path in glob.glob(folder + "pics/*"):
		store_path = "stored_image_embeddings_" + folder.split("/")[1] + "/" + folder_path.split('/')[-1] + ".pickle"
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
			print("Already preprocessed folder: %s" % folder_path, end="\r")


def embed_image(path):
	return predict(model, path)[0][0][0]


def print_corrupt_images():
	count = 0
	for folder_path in glob.glob("./train/pics/*"):
		for filepath in glob.glob(folder_path + "/*.jpg"):
			try:
				image.load_img(filepath, target_size=(224, 224))
			except:
				print("Warning, corrupt image: ", filepath)
				count += 1
	return count
