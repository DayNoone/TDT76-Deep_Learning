import pickle
import glob
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

# Get labels
train_labels_file = open("./../train/pickle/combined.pickle", 'rb')
train_labels = pickle.load(train_labels_file)
train_labels_file.close()

# Get stored labels embedding
f = open("labels_embedding.pickle", 'rb')
labels_embedding = pickle.load(f)
f.close()




def fetch_all_imagepaths():
	train_labels_file = open("./../train/pickle/combined.pickle", 'rb')
	train_labels = pickle.load(train_labels_file)
	train_labels_file.close()

	image_paths = []
	for filepath in glob.glob("./../train/pics/000000000/*.jpg"):
		image_id = get_id_from_path(filepath)
		if image_id in train_labels:
			image_paths.append(filepath)

	return image_paths


def get_id_from_path(file):
	image_id = (file.split(".jpg")[0]).split("/")[-1]
	return image_id


def check_max_min_similarity():
	paths = fetch_all_imagepaths()
	# max = 0
	# min = 1
	# for i in range(len(paths)):
	# 	image_1 = np.asarray(labels_embedding[get_id_from_path(paths[i])])
	# 	image_2 = np.asarray(labels_embedding[get_id_from_path(paths[i + 1])])
	# 	try:
	# 		sim = cosine_similarity([image_1], [image_2])[0][0]
	# 	except ValueError:
	# 		print("Error: ", sim)
	# 	if sim > max:
	# 		max = sim
	# 	if sim < min:
	# 		min = sim
	# print("Max: ", max)
	# print("Min: ", min)



# check_max_min_similarity()
def check_for_nan():
	print("Tot: ", len(labels_embedding))
	count = 0
	for label_key in labels_embedding:
		values = labels_embedding[label_key]
		for value in values:
			if math.isnan(value):
				count += 1
				break
	return count

# print(check_for_nan())
# check_max_min_similarity()