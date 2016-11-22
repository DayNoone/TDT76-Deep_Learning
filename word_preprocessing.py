#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np  # Make sure that numpy is imported
import pickle
import settings
from helpers.helpers import print_progress, load_pickle_file

"""
Modified code taken from my master project: https://github.com/ruoccoma/master_works/
"""


def get_relevant_word_embedding_dict(relevant_labels):
	if os.path.isfile("preprocessing/glove_embedding.pickle"):
		print("Fetching saved relevant word embedding dictionary")
		f = open("preprocessing/glove_embedding.pickle", 'rb')
		relevant_embedded_words = pickle.load(f)
		f.close()
	else:
		print("Creating relevant word embedding dictionary")
		count_read_glove_embedding = 0
		with open("preprocessing/glove.6B.300d.txt") as word_embedding:
			print("Getting word embeddings...")
			relevant_embedded_words = {}
			for line in word_embedding.readlines():
				line = (line.strip()).split(" ")
				embedded_word = line.pop(0)
				if embedded_word in relevant_labels:
					line = list(map(float, line))
					relevant_embedded_words[embedded_word] = np.asarray(line)
				if count_read_glove_embedding % 1000 == 0:
					print_progress(count_read_glove_embedding, 400000)
				count_read_glove_embedding += 1

		f = open("glove_embedding.pickle", 'wb')
		pickle.dump(relevant_embedded_words, f)
		f.close()
	return relevant_embedded_words


def get_relevant_words(labels_dict):
	relevant_words = []
	for label_key in labels_dict:
		for label, _ in labels_dict[label_key]:
			words = label.split(" ")
			if len(words) == 1:
				if words[0] not in relevant_words:
					relevant_words.append(words[0])
			else:
				for word in words:
					if word not in relevant_words:
						relevant_words.append(word)

	return relevant_words


def convert_sentence_to_vector(words, num_features, dictionary):
	# Function to average all of the word vectors in a given
	# paragraph
	#
	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	#
	nwords = 0.
	#
	# Loop over each word in the review and, if it is in the model's
	# vocaublary, add its feature vector to the total
	for word in words:
		# word_vector = fetch_word_vector(word, None)
		if word in dictionary:
			nwords += 1.
			featureVec = np.add(featureVec, dictionary[word])
	#
	# Divide the result by the number of words to get the average
	# if nwords == 0.:
	if nwords == 0.:
		return None
	featureVec = np.divide(featureVec, nwords)
	return featureVec


def convert_sentences(labels_dict, num_features):
	# Given a set of sentences (each one a list of words), calculate
	# the average feature vector for each one and return a 2D numpy array
	#
	# Initialize a counter
	counter = 0
	#
	# Preallocate a 2D numpy array, for speed
	len_sentences = len(labels_dict)
	sentenceFeatureVecs = {}

	word_embedding_dict = get_word_embedding_dict(labels_dict)

	print("Building word-vec dict complete.")
	# Loop through the words
	for label_key in labels_dict:
		if counter % 1000 == 0:
			print_progress(counter, len_sentences, prefix='Convert sentences:', suffix='Complete', bar_length=50)

		# Call the function (defined above) that makes average feature vectors
		words = []
		for label, confidence in labels_dict[label_key]:
			label = label.split(" ")
			for sub_label in label:
				words.append(sub_label)
		averaged_sentence_vector = convert_sentence_to_vector(words, num_features, word_embedding_dict)
		if averaged_sentence_vector is not None:
			sentenceFeatureVecs[label_key] = averaged_sentence_vector
		#
		# Increment the counter
		counter += 1
	print()

	word_embedding_file = open("preprocessing/labels_embedding.pickle", 'wb')
	pickle.dump(sentenceFeatureVecs, word_embedding_file, protocol=2)
	word_embedding_file.close()
	return sentenceFeatureVecs


def get_word_embedding_dict(labels_dict):
	if os.path.isfile("preprocessing/glove_embedding.pickle"):
		f = open("preprocessing/glove_embedding.pickle", 'rb')
		word_embedding_dict = pickle.load(f)
		f.close()
	else:
		relevant_words = get_relevant_words(labels_dict)
		word_embedding_dict = get_relevant_word_embedding_dict(relevant_words)
	return word_embedding_dict


def run_word_preprocessing(location="./train/"):
	if os.path.isfile(location + "pickle/combined.pickle"):
		train_labels = load_pickle_file(location + "pickle/combined.pickle")
	else:
		print("Missing combine.pickle for this dir")

	if os.path.isfile("preprocessing/labels_embedding.pickle"):
		f = open("preprocessing/labels_embedding.pickle", 'rb')
		labels_embedding = pickle.load(f)
		f.close()
	else:
		labels_embedding = convert_sentences(train_labels, settings.WORD_EMBEDDING_DIMENSION)
	return labels_embedding


if __name__ == "__main__":
	run_word_preprocessing()
