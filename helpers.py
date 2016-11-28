import glob
import sys
import math
import numpy as np
import pickle
import os
import tensorflow as tf
import keras


"""
Code taken from my master project: https://github.com/ruoccoma/master_works/
"""

def get_all_word_averaged_vectors():
	labels_embedding = load_pickle_file("preprocessing/labels_embedding.pickle")

	label_filenames = []
	label_vectors = []

	for folder_path in glob.glob("./preprocessing/stored_image_embeddings_train/*.pickle"):
		image_dictionary = load_pickle_file(folder_path)

		for image in image_dictionary:
			if image in labels_embedding:
				label_filenames.append(image)
				label_vectors.append(labels_embedding[image])
	return [label_filenames, label_vectors]

def get_all_trained_image_vectors():
	all_image_filnames = []
	all_image_vectors = []
	for folder_path in glob.glob("./preprocessing/trained_image_embeddings" + "/*.pickle"):
		image_folder_dictionary = load_pickle_file(folder_path)
		for image in image_folder_dictionary:
			all_image_filnames.append(image)
			all_image_vectors.append(image_folder_dictionary[image][0])
	return [all_image_filnames, all_image_vectors]

def get_all_image_vectors():
	all_image_filnames = []
	all_image_vectors = []
	for folder_path in glob.glob("./preprocessing/trained_image_embeddings" + "/*.pickle"):
		image_folder_dictionary = load_pickle_file(folder_path)
		for image in image_folder_dictionary:
			all_image_filnames.append(image)
			all_image_vectors.append(image_folder_dictionary[image][0])
	return [all_image_filnames, all_image_vectors]


def get_all_only_cnn_image_vectors():
	all_image_filnames = []
	all_image_vectors = []
	data_type = ("./train/").split("/")[1]
	for folder_path in glob.glob("./preprocessing/stored_image_embeddings_" + data_type + "/*.pickle"):
		image_folder_dictionary = load_pickle_file(folder_path)
		for image in image_folder_dictionary:
			all_image_filnames.append(image)
			all_image_vectors.append(image_folder_dictionary[image][0])
	return [all_image_filnames, all_image_vectors]





def load_pickle_file(path):
	if os.path.isfile(path):
		f = open(path, "rb")
		data = pickle.load(f)
		f.close()
		return data
	return None

def save_pickle_file(data, path):
	f = open(path, "wb")
	pickle.dump(data, f)
	f.close()


def l2norm(array):
	norm = math.sqrt(np.sum(([math.pow(x, 2) for x in array])))
	array = [x / norm for x in array]
	return array


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		barLength   - Optional  : character length of bar (Int)
	"""
	format_str = "{0:." + str(decimals) + "f}"
	percents = format_str.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	bar = '#' * filled_length + '-' * (bar_length - filled_length)
	sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

def tf_l2norm(tensor_array):
	norm = tf.sqrt(tf.reduce_sum(tf.pow(tensor_array, 2)))
	tensor_array /= norm
	return tensor_array

class WriteToFileCallback(keras.callbacks.Callback):
	def __init__(self, filename="training-epochs-results-DEFAULT.txt"):
		super(self.__class__, self).__init__()
		self.filename = filename

	def on_epoch_end(self, epoch, logs={}):
		file = open(self.filename, 'a')
		file.write("%s," % epoch)
		for k, v in logs.items():
			file.write("%s,%s," % (k, v))
		file.write("\n")
		file.close()