import pandas as pd
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

##WARNING! Some of the sentence-bert models will not load in properly. I think a problem with the torch/transformers install on my computer.

def process_labels(input_pickle_file, model_is_bert=True):
    # Returns the multi-word labels from the data as arrays of strings.
    # Sent bert model needs input as single multiword strings; word2vec needs arrays of single-word strings.

    object = pd.read_pickle(input_pickle_file)

    if model_is_bert:
        labels_list = object['class'].tolist()
    else:
        object[['class']] = object['class'].str.split()
        labels_list = object['class'].tolist()

    return labels_list


def get_mean_vector(word2vec_model, words):
    # Calculate mean vector of label.
    # source of this code: http://yaronvazana.com/2018/09/20/average-word-vectors-generate-document-paragraph-sentence-embeddings/
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []


def make_word2vec_file(filename, model, labels):
    # Get mean word2vec vector for all labels and write them to a file.

    kv = KeyedVectors(vector_size=model.wv.vector_size)
    vec_id_list = range(0, len(labels))

    vectors = []
    for label in labels:
        vec = get_mean_vector(model, label)
        vectors.append(vec)
    kv.add(vec_id_list, vectors)
    kv.save_word2vec_format(filename, binary=False)
    return

def make_word2vec_file_numpy(filename, model, labels):
    # Get mean word2vec vector for all labels and write them to a file. Labels are multi-word descriptions of a sound
    vectors = []
    for label in labels:
        vec = get_mean_vector(model, label)
        vectors.append(vec)
    embedding_np = np.array(vectors)

    with open(filename, 'wb') as f:
        np.save(f, embedding_np)
    return embedding_np


def get_sentence_bert(bert_sent_model, labels):
    # Get a single vector for each label which preserves semantic and contextual information.
    #add ability to change model
    # model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
    model = SentenceTransformer(bert_sent_model)
    sentence_embeddings = model.encode(labels)

    return sentence_embeddings


def make_bert_sentence_file(filename, bert_sent_model, labels, vec_size=300):
    #Get all the
    embeddings = get_sentence_bert(bert_sent_model, labels)
    kv = KeyedVectors(vector_size=vec_size)
    vec_id_list = range(0, len(labels))
    kv.add(vec_id_list, embeddings)
    kv.save_word2vec_format(filename, binary=False)
    return

def make_bert_sentence_file_numpy(filename, bert_sent_model, labels, vec_size=300):
    #Get all the
    embeddings = get_sentence_bert(bert_sent_model, labels)
    embedding_np = np.array(embeddings)
    with open(filename, 'wb') as f:
        np.save(f, embedding_np)
    return embedding_np


def open_numpy(file):
    with open(file, 'rb') as f:
        a = np.load(f)
    print(a)
    print(a.shape)
    return



def main():
    #toggle me!
    # train = False
    bert = False
    # bert = True
    train = True

    if train:
        INPUT_PATH = '/Users/samski/Downloads/train.pkl'

    else:
        INPUT_PATH = '/Users/samski/Downloads/test.pkl'



    if bert:
        if train:
            OUTFILE_PATH = 'bert_outfile_train.txt'
        else:
            OUTFILE_PATH = 'bert_outfile_test.txt'

        labels_list = process_labels(INPUT_PATH, model_is_bert=True)

        # # bert_sent_model = 'stsb-mpnet-base-v2' #doesn't work due to probable bad caching in torch!
        bert_sent_model = 'average_word_embeddings_glove.6B.300d'
        make_bert_sentence_file_numpy(OUTFILE_PATH, bert_sent_model, labels_list)
    else:
        if train:
            OUTFILE_PATH = 'word2vec_outfile_train_numpy.npy'
        else:
            OUTFILE_PATH = 'word2vec_outfile_test_numpy.npy'

        word2vec_model = api.load('glove-wiki-gigaword-300')
        labels_list = process_labels(INPUT_PATH, model_is_bert=False)
        make_word2vec_file_numpy(OUTFILE_PATH, word2vec_model, labels_list)



if __name__ == '__main__':
    main()