import glob
import os
import time
import numpy as np

from keras.callbacks import EarlyStopping
from keras.engine import Input, Model
from keras.layers import Dense

from word_preprocessing import run_word_preprocessing
from helpers import load_pickle_file, save_pickle_file, WriteToFileCallback, print_progress

custom_callback = WriteToFileCallback("results.txt")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

OPTIMIZER = "adam"
LOSS = "mse"
EPOCHS = 10
BATCH_SIZE = 128
MODEL_NAME = "MSE" + "-" + str(EPOCHS) + "-" + str(BATCH_SIZE)


def train_model(labels_embedding, location):
	if not model_is_saved():
		start_time = time.time()
		print("Start:", start_time)

		f = open("results.txt", 'a')

		f.write(50 * '=' + '\n' + MODEL_NAME + "\n")
		f.close()
		image_vectors, label_vectors = prepare_training_data(labels_embedding, location)
		image_vectors = np.asarray(image_vectors)
		label_vectors = np.asarray(label_vectors)

		model = get_base_model()

		model.compile(optimizer=OPTIMIZER, loss=LOSS)

		model.fit(image_vectors, label_vectors,
				  batch_size=BATCH_SIZE,
				  nb_epoch=EPOCHS,
				  callbacks=[custom_callback, early_stopping],
				  validation_split=0.1,
				  shuffle=True)
		print("End:", time.time() - start_time)
		save_model_to_file(model)

		save_trained_embeddings()


def get_base_model_large():
	image_inputs = Input(shape=(2048,), name="Image_input")
	image_model = Dense(2048, activation='relu')(image_inputs)
	image_model = Dense(1024, activation='relu')(image_model)
	image_model = Dense(1024, activation='relu')(image_model)
	image_model = Dense(1024, activation='relu')(image_model)
	image_model = Dense(1024, activation='relu')(image_model)
	image_model = Dense(512, activation='relu')(image_model)
	embedding_layer = Dense(512, activation='relu')(image_model)
	predictions = Dense(300, activation='relu')(embedding_layer)
	model = Model(input=image_inputs, output=predictions)
	return model

def get_base_model():
	image_inputs = Input(shape=(2048,), name="Image_input")
	image_model = Dense(1024, activation='relu')(image_inputs)
	image_model = Dense(1024, activation='relu')(image_model)
	image_model = Dense(512, activation='relu')(image_model)
	embedding_layer = Dense(512, activation='relu')(image_model)
	predictions = Dense(300, activation='relu')(embedding_layer)
	model = Model(input=image_inputs, output=predictions)
	return model

def save_trained_embeddings():
	model = load_model()

	start_time = time.time()
	count = 0
	tot = len(glob.glob("./stored_image_embeddings_train/*.pickle"))
	for file in glob.glob("./stored_image_embeddings_train/*.pickle"):
		store_path = "./trained_image_embeddings/" + file.split('/')[-1]
		trained_image_embeddings = {}
		image_dict = load_pickle_file(file)
		for image_filepath in image_dict:
			trained_image_embeddings[image_filepath] = model.predict(image_dict[image_filepath])
		save_pickle_file(trained_image_embeddings, store_path)
		print_progress(count, tot, prefix="Saving trained image embeddings")
		count += 1
	print("Time to save trained_embeddings: ", time.time() - start_time)


def predict_vector_on_model(vector, model):
	predicted_value = model.predict(np.array([vector]))
	return predicted_value


def prepare_training_data(labels_dictionary, location="./train/"):
	"""
	:param labels_dictionary: dictionary of labels (key: filename without .jpg, value: 300dim averaged label vector)
	:param location: ./train/ ./test/ ./validation/
	:return: All data to be trained on
	"""

	image_vectors = []
	label_vectors = []

	data_type = location.split("/")[1]
	for folder_path in glob.glob("./stored_image_embeddings_" + data_type + "/*.pickle"):
		image_dictionary = load_pickle_file(folder_path)

		for image in image_dictionary:
			if image in labels_dictionary:
				image_vectors.append(image_dictionary[image][0])
				label_vectors.append(labels_dictionary[image])
	return [image_vectors, label_vectors]


def model_is_saved():
	if os.path.isfile(MODEL_NAME + ".h5"):
		return True
	return False


def save_model_to_file(model):
	model.save_weights(MODEL_NAME + ".h5")
	print("Saved model \"%s\" to disk" % MODEL_NAME)


def load_model():
	model = get_base_model()
	print("Loading model \"%s\" from disk..." % MODEL_NAME)
	model.load_weights(MODEL_NAME + ".h5")
	model.compile(optimizer=OPTIMIZER, loss=LOSS)
	return model

if __name__ == "__main__":
	labels_dictionary = run_word_preprocessing("./train/")
	train_model(labels_dictionary, "./train/")
