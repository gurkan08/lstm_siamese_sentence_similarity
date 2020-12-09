
import fasttext
import pandas as pd
import os
from numpy import dot
from numpy.linalg import norm

model = fasttext.load_model("C:/Users/gurkan.sahin/Desktop/cc.tr.300.bin")
threshold = 0.5 # 0.48317757009345796
#threshold = 0.6 # 0.6383177570093458
#threshold = 0.7 # 0.7859813084112149
#threshold = 0.8 # 0.7598130841121495
#threshold = 0.9 # 0.6700934579439253

def _cosine_sim(a, b):
    #print(a.shape) # (300, )
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def fasttext_cosine_similarity(data):
    data["fasttext_cosine_similarity"] = [1 if _cosine_sim(model.get_sentence_vector(q1), model.get_sentence_vector(q2)) >= threshold
                                          else 0 for q1, q2 in zip(data["q1"], data["q2"])]
    data.to_excel(os.path.join("data", "test_fasttext_cosine_similarity.xlsx"), encoding="utf-8")
    print("test result: ", len(data[data["duplicate"] == data["fasttext_cosine_similarity"]]) / len(data))


if __name__ == '__main__':

    """
    test.xlsx preprocess edilmiş zaten !...
    """
    data = pd.read_excel(os.path.join("data", "test.xlsx"))
    fasttext_cosine_similarity(data)

