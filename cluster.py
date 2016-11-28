#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import glob

from sklearn.cluster import MiniBatchKMeans

from helpers import load_pickle_file, save_pickle_file, print_progress
from image_preprocessing import embed_image
from model import load_model, predict_vector_on_model


def get_cluster_members(trained_image_embedding):
	clustered_filenames = load_pickle_file("preprocessing/" + EVALUATION_FOLDER + "_clustered_filenames.pickle")
	cluster = get_cluster()
	predicted_cluster_id = cluster.predict(trained_image_embedding)
	cluster_member_filenames = []
	for i in range(len(cluster.labels_)):
		if predicted_cluster_id == cluster.labels_[i]:
			cluster_member_filenames.append(clustered_filenames[i].split("/")[-1])

	return cluster_member_filenames


def get_cluster():
	cluster_path = "preprocessing/" + EVALUATION_FOLDER + "_cluster.pickle"
	saved_cluster = load_pickle_file(cluster_path)
	if saved_cluster is not None:
		return saved_cluster
	return None


def create_cluster(vectors, cluster_path):
	saved_cluster = load_pickle_file(cluster_path)
	if saved_cluster is not None:
		print("Loading saved cluster...")
		return saved_cluster
	print("Creating cluster...")
	cluster = MiniBatchKMeans(n_clusters=100, random_state=0, init_size=3000)
	# cluster = DBSCAN(metric=cosine_similarity)
	cluster.fit(vectors)
	save_pickle_file(cluster, cluster_path)
	return cluster


def get_dict_cluster_sizes(cluster):
	cluster_dict = {}
	for id in cluster.labels_:
		if id in cluster_dict:
			count = cluster_dict[id]
			cluster_dict[id] = count + 1
		else:
			cluster_dict[id] = 1
	return cluster_dict

EVALUATION_FOLDER = "validate"

if __name__ == "__main__":
	# Model needs to be stored
	model = load_model()
	trained_image_filenames = []
	trained_image_embeddings = []
	count = 0
	for file in glob.glob("./validate/pics/*/*.jpg"):
		trained_image_filenames.append(file.split(".jpg")[0])
		image_embedding = embed_image(file)
		trained_image_embedding = predict_vector_on_model(image_embedding, model)[0]
		trained_image_embeddings.append(trained_image_embedding)
		print_progress(count + 1, 10000, prefix="Predicting images")
		count += 1
	print("Tr", len(trained_image_embeddings))
	save_pickle_file(trained_image_filenames, "preprocessing/" + EVALUATION_FOLDER + "_clustered_filenames.pickle")
	cluster = create_cluster(trained_image_embeddings, "preprocessing/" + EVALUATION_FOLDER + "_cluster.pickle")
	cluster_dict = get_dict_cluster_sizes(cluster)
	max_cluster_size = 0
	for i in cluster_dict:
		print("Cluster: ", i, " size: ", cluster_dict[i])
		if cluster_dict[i] > max_cluster_size:
			max_cluster_size = cluster_dict[i]
	print("Largest cluster: ", max_cluster_size)

	count = 0
	with open("cluster.csv", "w") as csvfile:
		wr = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, delimiter=";")
		for i in range(len(trained_image_filenames)):
			data = [trained_image_filenames[i].split("/")[-1], cluster.labels_[i], trained_image_embeddings[i]]
			wr.writerow(data)
			print_progress(count + 1, 10000, prefix="Writing to csv file")
			count += 1
