
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from numpy import dot
from numpy.linalg import norm
import numpy as np

threshold = 0.5 # 0.72
"""
threshold = 0.6 # 0.69
threshold = 0.7 # 0.66
threshold = 0.8 # 0.64
threshold = 0.9 # 0.61
"""

def _cosine_sim(a, b):
    #print(a.shape)
    #print(b.shape)
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def _train(data):
    X_train = []
    X_train.extend(data['q1'].tolist())
    X_train.extend(data['q2'].tolist())

    # tf-idf matrix for train
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)  # fit_transform
    tfidf_transformer = TfidfTransformer(use_idf=True)  # fit_transform
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return count_vect, tfidf_transformer

def _test(count_vect, tfidf_transformer, test_data):
    results = []
    for q1, q2 in zip(test_data["q1"], test_data["q2"]):
        q1_counts = count_vect.transform([q1])  # not fit_transform, transform
        q1_tfidf = tfidf_transformer.transform(q1_counts)  # not fit_transform, transform

        q2_counts = count_vect.transform([q2])
        q2_tfidf = tfidf_transformer.transform(q2_counts)

        tfidf_cosine_similarity = _cosine_sim(np.asarray(q1_tfidf.todense()).ravel(),
                                              np.asarray(q2_tfidf.todense()).ravel())
        if tfidf_cosine_similarity >= threshold:
            results.append(1)
        else:
            results.append(0)

    test_data["tfidf_similarity"] = results
    test_data.to_excel(os.path.join("data", "test_tfidf_cosine_similarity.xlsx"), encoding="utf-8")
    print("test result (tfidf similarity): ", len(test_data[test_data["duplicate"] == test_data["tfidf_similarity"]])
          / len(test_data))

if __name__ == '__main__':

    """
    train_data / test_data preprocess edilmiş...!
    """
    train_data = pd.read_excel(os.path.join("data", "train.xlsx"))
    test_data = pd.read_excel(os.path.join("data", "test.xlsx"))

    count_vect, tfidf_transformer = _train(train_data)
    _test(count_vect, tfidf_transformer, test_data)

