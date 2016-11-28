from sklearn.metrics.pairwise import cosine_similarity

import model
from cluster import get_cluster_members
from helpers import print_progress, get_all_trained_image_vectors
from image_preprocessing import embed_image
from model import predict_vector_on_model, load_model
from word_preprocessing import *

USE_CLUSTERING = False

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
    model = load_model()


    for query in queries:
        query_path = get_file_from_path(query + ".jpg", location)
        image_embedding = embed_image(query_path)
        trained_image_embedding = predict_vector_on_model(image_embedding, model)
        if USE_CLUSTERING:
            most_similar = get_cluster_members(trained_image_embedding)
        else:
            most_similar = get_most_similar(trained_image_embedding)
        my_return_dict[query] = most_similar
        print_progress(count, tot, prefix="Retrieving similar images", suffix=len(most_similar))
        count += 1
    return my_return_dict


def get_most_similar(trained_image_embedding):
    trained_image_filenames, trained_image_vectors = get_all_trained_image_vectors()
    cos_values = cosine_similarity(trained_image_embedding, trained_image_vectors)[0]
    similarities = []
    for i in range(len(trained_image_filenames)):
        similarities.append((trained_image_filenames[i], cos_values[i]))
    similarities.sort(key=lambda s: s[1], reverse=True)

    most_similar_filenames = []
    for tuple in similarities:
        most_similar_filenames.append(tuple[0])
        if tuple[1] < 0.9:
            break
    return most_similar_filenames


def get_file_from_path(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
