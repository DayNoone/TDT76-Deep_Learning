#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import time
import numpy as np
from helpers.helpers import load_pickle_file, save_pickle_file, get_all_image_vectors


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




def get_cluster(vectors):
	cluster_path = "preprocessing/image_vector_cluster.pickle"
	saved_cluster = load_pickle_file(cluster_path)
	if saved_cluster is not None:
		return saved_cluster
	vectors = np.asarray(vectors)
	cluster = MiniBatchKMeans(n_clusters=1000, random_state=0, init_size=3000)
	cluster.fit(vectors)
	save_pickle_file(cluster, cluster_path)
	return cluster


def find_most_similar_in_dataset(target, location, k=1):
	data_type = location.split("/")[1]
	for folder_path in glob.glob("./preprocessing/stored_image_embeddings_" + data_type + "/*.pickle"):
		image_dictionary = open_file(folder_path)

		for image in image_dictionary:
			cosine_similarity([target], [image_dictionary[image]])


def find_most_similar_in_collection(target_vector, vector_names, vectors, k=1):
	most_similar = -1
	most_similar_filename = "All_less_than_-1"
	for i in range(len(vectors)):
		sim = cosine_similarity([target_vector], [vectors[i]])
		if sim > most_similar:
			most_similar = sim
			most_similar_filename = vector_names[i]
	return [most_similar_filename], [most_similar]


def evaluate():
	start_time = time.time()
	f = open("./preprocessing/stored_image_embeddings_train/000000000.pickle", "rb")
	image_dictionary = pickle.load(f)
	f.close()
	for image in image_dictionary:
		find_most_similar_in_dataset(image_dictionary[image], "./train/")
		break


def get_predicted_cluster_id(cluster, vector):
	return cluster.predict([vector])


def get_cluster_members(cluster_id, cluster, image_filenames, image_vectors):
	cluster_member_filenames = []
	cluster_member_vectors = []
	for i in range(len(cluster.labels_)):
		if cluster_id == cluster.labels_[i]:
			cluster_member_filenames.append(image_filenames[i])
			cluster_member_vectors.append(image_vectors[i])
	return [cluster_member_filenames, cluster_member_vectors]


def compare_to_cluster(vector):
	all_image_filenames, all_image_vectors = get_all_image_vectors()
	image_cluster = get_cluster(all_image_vectors)
	predicted_cluster_id = get_predicted_cluster_id(image_cluster, vector)
	predicted_cluster_member_filenames, predicted_cluster_member_vectors = get_cluster_members(predicted_cluster_id,
																							   image_cluster,
																							   all_image_filenames,
																							   all_image_vectors)
	most_similar_filename, most_similar_value = find_most_similar_in_collection(vector,
																				predicted_cluster_member_filenames,
																				predicted_cluster_member_vectors)
	return (most_similar_filename, most_similar_value)


if __name__ == "__main__":
	start_time = time.time()
	# evaluate()
	all_image_filenames, all_image_vectors = get_all_only_cnn_image_vectors()
	image_cluster = get_cluster(all_image_vectors)

	image_no = 0
	predicted_cluster_id = get_predicted_cluster_id(image_cluster, all_image_vectors[image_no])
	predicted_cluster_member_filenames, predicted_cluster_member_vectors = get_cluster_members(predicted_cluster_id, image_cluster, all_image_filenames, all_image_vectors)
	most_similar_filename, most_similar_value = find_most_similar_in_collection(all_image_vectors[image_no], predicted_cluster_member_filenames, predicted_cluster_member_vectors)
	# print("Size of predicted cluster: ", len(predicted_cluster_member_filenames))
	# print("Target image: ", all_image_filenames[image_no], all_image_vectors[image_no])
	# print("Most similar image: ", most_similar_filename, most_similar_value)
	print("Time: ", time.time() - start_time)

