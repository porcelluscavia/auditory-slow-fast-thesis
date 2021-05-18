import pandas as pd
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors




def get_mean_vector(word2vec_model, words):
    # source of this code: http://yaronvazana.com/2018/09/20/average-word-vectors-generate-document-paragraph-sentence-embeddings/
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []





def main():
    DATA_PATH = '/Users/samski/Downloads/test.pkl'
    OUTFILE_PATH = "word2vec_outfile_test.txt"

    model = api.load("glove-wiki-gigaword-100")

    object = pd.read_pickle(r'/Users/samski/Downloads/test.pkl')

    object[['class']] = object['class'].str.split()
    desc_list = object['class'].tolist()

    kv = KeyedVectors(vector_size=model.wv.vector_size)

    vec_id_list = range(0, len(desc_list))


    vectors = []
    for label in desc_list:
        vec = get_mean_vector(model, label)
        vectors.append(vec)

    kv.add(vec_id_list, vectors)
    kv.save_word2vec_format(OUTFILE_PATH, binary=False)



if __name__ == '__main__':
    main()