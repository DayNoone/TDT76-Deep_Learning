import glob

import time
from keras.layers import Dense, Dropout, Lambda
from keras.engine import Input, Model
import keras.backend as K

from keras import callbacks
import numpy as np
import pickle
import os

# from custom_callback import WriteToFileCallback
# custom_callback = WriteToFileCallback(settings.RESULT_TEXTFILE_PATH)
from word_preprocessing import *
from helpers.helpers import load_pickle_file, get_all_image_vectors, save_pickle_file

OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"
EPOCHS = 30
BATCH_SIZE = 128
MODEL_NAME = "FF" + "-" + str(EPOCHS) + "-" + str(BATCH_SIZE)



def train_model(labels_embedding, location):
	if model_is_saved():
		model = load_model()
	else:
		image_vectors, label_vectors = prepare_training_data(labels_embedding, location)
		image_vectors = np.asarray(image_vectors)
		label_vectors = np.asarray(label_vectors)

		model = get_base_model()

		model.compile(optimizer=OPTIMIZER, loss=LOSS)

		model.fit(image_vectors, label_vectors,
				  batch_size=BATCH_SIZE,
				  nb_epoch=EPOCHS,
				  shuffle=True,
				  validation_split=0.2)

		save_model_to_file(model)
	save_trained_embeddings()


def get_base_model():
	image_inputs = Input(shape=(2048,), name="Image_input")
	image_model = Lambda(lambda x: abs(x), name="Image_abs_1")(image_inputs)
	image_model = Dense(1024, activation='relu')(image_model)
	embedding_layer = Dense(512, activation='relu', name="embedding_layer")(image_model)
	predictions = Dense(300, activation='softmax', name="softmax_layer")(embedding_layer)
	model = Model(input=image_inputs, output=predictions)
	return model


def get_prediction_model():
	output_layer = "embedding_layer"
	base_model = load_model()
	model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
	model.compile(optimizer=OPTIMIZER, loss=LOSS)
	return model


def save_trained_embeddings():
	model = get_prediction_model()

	start_time = time.time()
	count = 0
	tot = len(glob.glob("./preprocessing/stored_image_embeddings_train/*.pickle"))
	for file in glob.glob("./preprocessing/stored_image_embeddings_train/*.pickle"):
		store_path = "./preprocessing/trained_image_embeddings/" + file.split('/')[-1]
		trained_image_embeddings = {}
		if not os.path.isfile(store_path):
			image_dict = load_pickle_file(file)
			print(store_path)
			for image_filepath in image_dict:
				trained_image_embeddings[image_filepath] = model.predict(image_dict[image_filepath])
			save_pickle_file(trained_image_embeddings, store_path)

		else:
			print("Skipping already created file", store_path)
		print_progress(count, tot, prefix="Saving trained image embeddings")
		count += 1
	print("Time to save trained_embeddings: ", start_time - time.time())


def model_is_saved():
	if os.path.isfile("stored_models/" + MODEL_NAME + ".h5"):
		return True
	return False


def save_model_to_file(model):
	model.save_weights("stored_models/" + MODEL_NAME + ".h5")
	print("Saved model \"%s\" to disk" % MODEL_NAME)


def load_model():
	model = get_base_model()
	print("Loading model \"%s\" from disk..." % MODEL_NAME)
	model.load_weights("stored_models/" + MODEL_NAME + ".h5")
	model.compile(optimizer=OPTIMIZER, loss=LOSS)
	return model


def prepare_training_data(labels_dictionary, location="./train/"):
	"""
	:param labels_dictionary: dictionary of labels (key: filename without .jpg, value: 300dim averaged label vector)
	:param location: ./train/ ./test/ ./validation/
	:return: All data to be trained on
	"""

	image_vectors = []
	label_vectors = []

	data_type = location.split("/")[1]
	for folder_path in glob.glob("./preprocessing/stored_image_embeddings_" + data_type + "/*.pickle"):
		image_dictionary = load_pickle_file(folder_path)

		for image in image_dictionary:
			if image in labels_dictionary:
				image_vectors.append(image_dictionary[image][0])
				label_vectors.append(labels_dictionary[image])
	return [image_vectors, label_vectors]


if __name__ == "__main__":
	labels_dictionary = run_word_preprocessing("./train/")
	train_model(labels_dictionary, "./train/")

