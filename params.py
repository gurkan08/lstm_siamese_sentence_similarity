
import os
import fasttext

class Params(object):

    fasttext_model = fasttext.load_model("C:/Users/gurkan.sahin/Desktop/cc.tr.300.bin")

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    preprocess_steps = {
        "lowercase": True,
        "remove_punctuations": True,
        "remove_numbers": True,
        "remove_stop_words": False,
        "zemberek_stemming": False,
        "first_5_char_stemming": False,
        "data_shuffle": False
    }

    data_dir = "data/TurkQP.csv"
    test_size = 0.3
    valid_size = 0.1
    random_state = 42

    fasttext_embedding_use = True # True :)
    if fasttext_embedding_use:
        embed_size = 300
        embedding_matrix = None
    else:
        embed_size = 150
    threshold = 0.5 # 0.5
    epoch = 20
    batch_size = 32
    lr = 1e-3
    max_sentence_size = 100 # manuel
    lstm_units = 100
    dense_size = 50
    label_size = 2
    drop_out = 0.3

    vocab_size = None
    sentence_tokenizer = None
    sentence_tokenizer_name = "sentence_tokenizer.pickle"
    optimizer = "adam"
    early_stop_patience = 3
    ReduceLROnPlateau_factor = 0.9  # 0.1 çok fazla azaltıyor ..! # new_lr = lr * factor
    ReduceLROnPlateau_patience = 2
    ReduceLROnPlateau_min_lr = 1e-6

