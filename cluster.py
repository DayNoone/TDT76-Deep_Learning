#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import numpy as np
from helpers.helpers import load_pickle_file, save_pickle_file, get_all_word_embeddings

CLUSTER_PATH = "preprocessing/word_embedding_cluster.pickle"

def get_cluster():
	saved_cluster = load_pickle_file(CLUSTER_PATH)
	if saved_cluster is not None:
		return saved_cluster
	return None


def create_cluster(vectors):
	saved_cluster = load_pickle_file(CLUSTER_PATH)
	if saved_cluster is not None:
		print("Loading saved cluster...")
		return saved_cluster
	print("Creating cluster...")
	norm_vectors = normalize(vectors, return_norm=True)[0]
	print("Lenght of clustered vectors", len(vectors[0]))
	cluster = MiniBatchKMeans(n_clusters=100, random_state=0, init_size=3000)
	# cluster = DBSCAN(metric=cosine_similarity)
	cluster.fit(vectors)
	save_pickle_file(cluster, CLUSTER_PATH)
	return cluster


def compare_to_cluster(vector, image_cluster, k):
	vector = normalize(vector)
	all_filenames, all_embedded_vectors = get_all_word_embeddings()
	predicted_cluster_id = image_cluster.predict(vector)

	cluster_member_filenames = []
	cluster_member_vectors = []
	for i in range(len(image_cluster.labels_)):
		if predicted_cluster_id == image_cluster.labels_[i]:
			cluster_member_filenames.append(all_filenames[i])
			cluster_member_vectors.append(all_embedded_vectors[i])
	print("Cluster size: ", len(cluster_member_filenames))
	similarities = []
	for i in range(len(cluster_member_filenames)):
		similarity = cosine_similarity(vector, [cluster_member_vectors[i]])
		similarities.append((cluster_member_filenames[i], similarity))
	similarities.sort(key=lambda s: s[1], reverse=True)
	most_similar_filenames = []
	for tuple in similarities[:k]:
		most_similar_filenames.append(tuple[0])
	return most_similar_filenames, predicted_cluster_id


def get_dict_cluster_sizes(cluster):
	cluster_dict = {}
	for id in cluster.labels_:
		if id in cluster_dict:
			count = cluster_dict[id]
			cluster_dict[id] = count + 1
		else:
			cluster_dict[id] = 1
	return cluster_dict

if __name__ == "__main__":
	all_filenames, all_embedded_vectors = get_all_word_embeddings()
	cluster = create_cluster(all_embedded_vectors)
	print("Query: ", all_filenames[2])
	most_similar_filenames = compare_to_cluster([all_embedded_vectors[2]], cluster, 3)
	print("Most similar: ", most_similar_filenames)
