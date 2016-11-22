import glob
import pickle
import random

import time
from PIL import Image

from word_preprocessing import *
import model
from cluster import compare_to_cluster, create_cluster, get_cluster, get_dict_cluster_sizes
from image_preprocessing import embed_image
from helpers.helpers import load_pickle_file, get_all_word_embeddings
from model import predict_vector_on_model, load_model


def train(location='./train/'):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
    must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
    :param location: The location of the training data folder hierarchy
    :return: nothing
    """

    # run_vgg(location)
    labels_embedding = run_word_preprocessing()
    model.train_model(labels_embedding, location)
    all_filenames, all_embedded_vectors = get_all_word_embeddings()
    cluster = create_cluster(all_embedded_vectors)
    # cluster_dict = get_dict_cluster_sizes(cluster)
    # for i in cluster_dict:
    #     print("Cluster id: ", i, " Size: ", cluster_dict[i])

def test(queries=list(), location='./test'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned
    :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
    with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
    :param location: The location of the test data folder hierarchy
    :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
    retrieved for that input
    """

    # ##### The following is an example implementation -- that would lead to 0 points  in the evaluation :-)
    my_return_dict = {}

    # Load the dictionary with all training files. This is just to get a hold of which
    # IDs are there; will choose randomly among them
    # training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
    # training_labels = list(training_labels.keys())
    count = 0
    tot = len(queries)
    cluster = get_cluster()
    model = load_model()
    for query in queries:

        # This is the image. Just opening if here for the fun of it; not used later
        # query_image = Image.open(location + '/pics/' + query + '.jpg')
        # query_image.show()

        # Generate a random list of 50 entries
        # cluster = [training_labels[random.randint(0, len(training_labels) - 1)] for idx in range(50)]
        image_embedding = embed_image(location + '/pics/' + query + '.jpg')
        trained_image_embedding = predict_vector_on_model(image_embedding, model)
        print("Trained img embedding size: ", len(trained_image_embedding[0]))
        cluster_filenames, cluster_id = compare_to_cluster(trained_image_embedding, cluster, 50)
        print("Cluster filenames", cluster_filenames)
        my_return_dict[query] = cluster_filenames
        print_progress(count, tot, prefix="Predicting images")
        count += 1
    return my_return_dict

if __name__ == "__main__":
    start_time = time.time()
    train()
    labels_dict = load_pickle_file("./validate/pickle/descriptions000000000.pickle")
    predicted_images_dict = test([(f.split("pics/")[-1]).split(".jpg")[0] for f in glob.glob("./validate/pics/000000000/*.jpg")], "./validate")
    print("Time: ", time.time() - start_time)
