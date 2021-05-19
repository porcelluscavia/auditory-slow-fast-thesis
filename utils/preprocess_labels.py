import pandas as pd
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer


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


def get_sentence_bert(bert_sent_model, labels):
    # Get a single vector for each label which preserves semantic and contextual information.

    bert_sent_model = SentenceTransformer(bert_sent_model)
    sentence_embeddings = bert_sent_model.encode(labels)
    return sentence_embeddings

def make_bert_sentence_file(filename, bert_sent_model, labels, vec_size=100):
    embeddings = get_sentence_bert(labels)
    kv = KeyedVectors(vector_size=vec_size)
    vec_id_list = range(0, len(labels))

    kv.add(vec_id_list, embeddings)
    kv.save_word2vec_format(filename, binary=False)
    return



def main():
    # INPUT_PATH = '/Users/samski/Downloads/test.pkl'
    # OUTFILE_PATH = 'word2vec_outfile_test.txt'
    INPUT_PATH = '/Users/samski/Downloads/train.pkl'
    OUTFILE_PATH = 'word2vec_outfile_train.txt'

    word2vec_model = api.load('glove-wiki-gigaword-100')
    bert_sent_model = 'stsb-mpnet-base-v2'

    object = pd.read_pickle(INPUT_PATH)

    object[['class']] = object['class'].str.split()
    labels_list = object['class'].tolist()

    make_word2vec_file(OUTFILE_PATH, word2vec_model, labels_list)

    embedding = get_sentence_bert(bert_sent_model, labels_list[0])
    print(embedding.shape)




if __name__ == '__main__':
    main()