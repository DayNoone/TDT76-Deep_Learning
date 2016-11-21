import glob
import sys
import math
import numpy as np
import pickle
import os


"""
Code taken from my master project: https://github.com/ruoccoma/master_works/
"""

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


def get_all_trained_image_vectors():
	all_image_filnames = []
	all_image_vectors = []
	for folder_path in glob.glob("./preprocessing/trained_image_embeddings" + "/*.pickle"):
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
