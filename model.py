import glob

from keras.layers import Dense, Dropout, Lambda, BatchNormalization
from keras.engine import Input, Model
from keras.callbacks import EarlyStopping

from helpers.helpers import WriteToFileCallback
from word_preprocessing import *
from helpers.helpers import load_pickle_file
from sklearn.preprocessing import normalize

OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"
EPOCHS = 50
BATCH_SIZE = 128
MODEL_NAME = "FF" + "-" + str(EPOCHS) + "-" + str(BATCH_SIZE)
custom_callback = WriteToFileCallback("results.txt")
early_stopping = EarlyStopping(monitor='val_loss', patience=5)


def train_model(labels_embedding, location):
	if not model_is_saved():
		image_vectors, label_vectors = prepare_training_data(labels_embedding, location)
		image_vectors = np.array(image_vectors)
		label_vectors = normalize(label_vectors)

		model = get_base_model()

		model.compile(optimizer=OPTIMIZER, loss=LOSS)

		model.fit(image_vectors, label_vectors,
				  batch_size=BATCH_SIZE,
				  nb_epoch=EPOCHS,
				  shuffle=True,
				  callbacks=[custom_callback, early_stopping],
				  validation_split=0.1)

		save_model_to_file(model)


def get_base_model():
	image_inputs = Input(shape=(2048,), name="Image_input")
	image_model = BatchNormalization()(image_inputs)
	image_model = Dense(1024, activation='relu')(image_model)
	image_model = BatchNormalization()(image_model)
	image_model = Dense(512, activation='relu')(image_model)
	image_model = Dropout(0.2)(image_model)
	image_model = BatchNormalization()(image_model)
	predictions = Dense(300, activation='softmax', name="softmax_layer")(image_model)
	model = Model(input=image_inputs, output=predictions)
	return model


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
	for folder_path in glob.glob("./preprocessing/stored_image_embeddings_" + data_type + "/*.pickle"):
		image_dictionary = load_pickle_file(folder_path)

		for image in image_dictionary:
			if image in labels_dictionary:
				image_vectors.append(image_dictionary[image][0])
				label_vectors.append(labels_dictionary[image])
	return [image_vectors, label_vectors]


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

if __name__ == "__main__":
	labels_dictionary = run_word_preprocessing("./train/")
	train_model(labels_dictionary, "./train/")
