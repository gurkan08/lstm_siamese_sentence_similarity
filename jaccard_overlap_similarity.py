
import os
import pandas as pd
import requests
import ast

threshold = 0.5 # 0.75
"""
threshold = 0.6 # 0.69
threshold = 0.7 # 0.66
threshold = 0.8 # 0.63
threshold = 0.9 # 0.61
"""

def zemberek_stemming(data):
    API_ENDPOINT = "http://localhost:4567/stems"
    for id, row in data.iterrows():
        # for q1
        stemmed_word_list = []
        for word in row["q1"].split():
            _data = {
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "word": word.rstrip()
            }
            result = requests.post(url=API_ENDPOINT, data=_data)
            result = result.content.decode("UTF-8")
            result = ast.literal_eval(result)
            # print(result)
            if len(result["results"]):
                stemmed_word_list.append(result["results"][0]["stems"][0])
            else:  # add original word
                stemmed_word_list.append(word.rstrip())
        data.loc[id, "q1"] = " ".join(stemmed_word_list)

        # for q2
        stemmed_word_list = []
        for word in row["q2"].split():
            _data = {
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "word": word.rstrip()
            }
            result = requests.post(url=API_ENDPOINT, data=_data)
            result = result.content.decode("UTF-8")
            result = ast.literal_eval(result)
            # print(result)
            if len(result["results"]):
                stemmed_word_list.append(result["results"][0]["stems"][0])
            else: # add original word
                stemmed_word_list.append(word.rstrip())
        data.loc[id, "q2"] = " ".join(stemmed_word_list)

    return data


def jaccard_similarity(data):
    results = []
    for q1, q2 in zip(data["q1"], data["q2"]):
        q1_unique_words = set(q1.split())
        q2_unique_words = set(q2.split())
        intersection = len(q1_unique_words.intersection(q2_unique_words))
        union = len(q1_unique_words) + len(q2_unique_words) - intersection
        jaccard_score = intersection / union
        if jaccard_score >= threshold:
            results.append(1)
        else:
            results.append(0)

    data["jaccard_similarity"] = results
    data.to_excel(os.path.join("data", "test_jaccard_similarity.xlsx"), encoding="utf-8")
    print("test result (jaccard similarity): ", len(data[data["duplicate"] == data["jaccard_similarity"]]) / len(data))


def overlap_similarity(data):
    results = []
    for q1, q2 in zip(data["q1"], data["q2"]):
        q1_unique_words = set(q1.split())
        q2_unique_words = set(q2.split())
        intersection = len(q1_unique_words.intersection(q2_unique_words))
        min_len = min(len(q1_unique_words), len(q2_unique_words))
        overlap_score = intersection / min_len
        if overlap_score >= threshold:
            results.append(1)
        else:
            results.append(0)

    data["overlap_similarity"] = results
    data.to_excel(os.path.join("data", "test_overlap_similarity.xlsx"), encoding="utf-8")
    print("test result (overlap similarity): ", len(data[data["duplicate"] == data["overlap_similarity"]]) / len(data))


if __name__ == '__main__':

    """
    test.xlsx preprocess edilmiş zaten !...
    """

    """
    # create stemmed test.xlsx
    data = pd.read_excel(os.path.join("data", "test.xlsx"))
    data = zemberek_stemming(data)
    # save stemmed file
    data.to_excel(os.path.join("data", "test_stemming.xlsx"), encoding="utf-8")
    """

    data = pd.read_excel(os.path.join("data", "test_stemming.xlsx"))
    jaccard_similarity(data)
    overlap_similarity(data)

