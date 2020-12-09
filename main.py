
"""
kaynak: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.autograd import Variable
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

from params import Params
from preprocess import *
from model import LSTMModel

class Main(object):

    def __init__(self):
        pass

    @staticmethod
    def read_dataset():
        data = pd.read_csv(Params.data_dir, encoding="utf-8")
        return data

    @staticmethod
    def do_preprocess(data):
        if Params.preprocess_steps["lowercase"]:
            data = lowercase(data)
        if Params.preprocess_steps["remove_punctuations"]:
            data = remove_punctuations(data)
        if Params.preprocess_steps["remove_numbers"]:
            data = remove_numbers(data)
        return data

    @staticmethod
    def _train_test_split(df):
        train, test = train_test_split(df,
                                       stratify=df[["duplicate"]],
                                       test_size=Params.test_size,
                                       random_state=Params.random_state)
        return train, test

    @staticmethod
    def tokenize_and_texts_to_sequences(train_df, test_df):
        # merge q1 & q2 on train to create vocab dict
        train_df['q1_q2'] = train_df[['q1', 'q2']].apply(lambda x: ' '.join(x), axis=1)
        """
        # ok
        print(train_df["q1"][0])
        print(train_df["q2"][0])
        print(train_df["q1_q2"][0])
        """
        Params.sentence_tokenizer = Tokenizer(oov_token="UNK",
                                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                              lower=True)  # 0 index reserved as padding_value
        Params.sentence_tokenizer.fit_on_texts(train_df["q1_q2"])
        Params.vocab_size = len(Params.sentence_tokenizer.word_index) + 1
        #print("---> ", Params.vocab_size)

        q1_train_sentences = Params.sentence_tokenizer.texts_to_sequences(train_df["q1"])  # list
        q2_train_sentences = Params.sentence_tokenizer.texts_to_sequences(train_df["q2"])
        q1_train_sentences = pad_sequences(q1_train_sentences, maxlen=Params.max_sentence_size, padding="post", value=0.)
        q2_train_sentences = pad_sequences(q2_train_sentences, maxlen=Params.max_sentence_size, padding="post", value=0.)
        new_train_df = pd.DataFrame(zip(q1_train_sentences, q2_train_sentences, train_df["duplicate"].tolist()),
                                    columns=["q1", "q2", "duplicate"])
        with open(os.path.join(Params.model_dir, Params.sentence_tokenizer_name), "wb") as handle:
            pickle.dump(Params.sentence_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # test
        q1_test_sentences = Params.sentence_tokenizer.texts_to_sequences(test_df["q1"])
        q2_test_sentences = Params.sentence_tokenizer.texts_to_sequences(test_df["q2"])
        q1_test_sentences = pad_sequences(q1_test_sentences, maxlen=Params.max_sentence_size, padding="post", value=0.)
        q2_test_sentences = pad_sequences(q2_test_sentences, maxlen=Params.max_sentence_size, padding="post", value=0.)
        new_test_df = pd.DataFrame(zip(q1_test_sentences, q2_test_sentences, test_df["duplicate"].tolist()),
                                   columns=["q1", "q2", "duplicate"])
        return new_train_df, new_test_df

    @staticmethod
    def fasttext_embedding_init():
        # keras embeddings_initializer = 'uniform'
        embedding_matrix = np.zeros((len(Params.sentence_tokenizer.word_index) + 1, Params.embed_size))  # +1:zero_pad
        embedding_matrix[0, :] = np.random.uniform(size=(Params.embed_size,))  # 0 index for padding_value
        # for other vocab words, (UNK 1), 1. index == UNK word !
        for key, value in Params.sentence_tokenizer.word_index.items():
            embedding_matrix[value, :] = Params.fasttext_model.get_word_vector(key)
        Params.embedding_matrix = embedding_matrix  # update embedding_matrix

    @staticmethod
    def run_train(train_df):
        # init fasttext embedding weights
        Main.fasttext_embedding_init()

        model_obj = LSTMModel(max_sentence_size=Params.max_sentence_size,
                              embed_size=Params.embed_size,
                              vocab_size=len(Params.sentence_tokenizer.word_index) + 1,
                              lstm_units=Params.lstm_units,
                              dense_size=Params.dense_size,
                              label_size=Params.label_size)
        model = model_obj.get_model()

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=Params.ReduceLROnPlateau_factor,
                                                         patience=Params.ReduceLROnPlateau_patience,
                                                         min_lr=Params.ReduceLROnPlateau_min_lr)

        if Params.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=Params.lr)
        elif Params.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=Params.lr, beta_1=0.9, beta_2=0.999)

        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])
        print("------------model summary-------------")
        print(model.summary())

        # split train-valid
        # validation_split=Params.validation_split # dataset sonundan % x'i valid olarak alıyor, yanlış yöntem !
        train, valid = train_test_split(train_df,
                                        stratify=train_df[["duplicate"]],
                                        test_size=Params.test_size,
                                        random_state=Params.random_state)

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=Params.early_stop_patience),
            reduce_lr
        ]
        history = model.fit([np.array(train["q1"].tolist()), np.array(train["q2"].tolist())],
                            np.array(train["duplicate"].tolist()),
                            batch_size=Params.batch_size,
                            epochs=Params.epoch,
                            validation_data=([np.array(valid["q1"].tolist()), np.array(valid["q2"].tolist())],
                                             np.array(valid["duplicate"].tolist())),
                            verbose=1,
                            shuffle=True,
                            callbacks=my_callbacks)
        model.save(os.path.join(Params.model_dir, "model.h5"))

        print("-------history---------")
        print(history.history)

        Main.plot(history)
        return model

    @staticmethod
    def plot(model):
        # Plot accuracy
        plt.clf()
        plt.plot(model.history['acc'])
        plt.plot(model.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(Params.plot_dir, "accuracy.png"))

        # Plot loss
        plt.clf()
        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        #plt.show()
        plt.savefig(os.path.join(Params.plot_dir, "loss.png"))

    @staticmethod
    def run_test(model, test_df):
        loss_metrics = model.evaluate([np.array(test_df["q1"].tolist()), np.array(test_df["q2"].tolist())],
                                      np.array(test_df["duplicate"].tolist()))
        return loss_metrics

    @staticmethod
    def run_test_2(model):
        """
        predict api...on test set
        """
        data = pd.read_excel(os.path.join("data", "test.xlsx"))
        data = Main.do_preprocess(data)
        q1 = Params.sentence_tokenizer.texts_to_sequences(data["q1"])
        q2 = Params.sentence_tokenizer.texts_to_sequences(data["q2"])
        q1 = pad_sequences(q1, maxlen=Params.max_sentence_size, padding="post", value=0.)
        q2 = pad_sequences(q2, maxlen=Params.max_sentence_size, padding="post", value=0.)
        test_df = pd.DataFrame(zip(q1, q2, data["duplicate"].tolist()),
                               columns=["q1", "q2", "duplicate"])
        result = model.predict([np.array(test_df["q1"].tolist()), np.array(test_df["q2"].tolist())])
        print(result)
        predict_list = []
        for prob in result:
            if prob >= Params.threshold:
                predict_list.append(1)
            else:
                predict_list.append(0)
        data["siamese_result"] = pd.Series(predict_list)
        data.to_excel(os.path.join("data", "test_siamese_similarity.xlsx"), encoding="utf-8")
        print("test result: ", len(data[data["duplicate"] == data["siamese_result"]]) / len(data))


if __name__ == '__main__':

    data = Main.read_dataset()
    #print(data)
    print(data.columns)

    data = Main.do_preprocess(data)

    train_df, test_df = Main._train_test_split(data)
    train_df.to_excel(os.path.join("data", "train.xlsx"), encoding="utf-8")
    test_df.to_excel(os.path.join("data", "test.xlsx"), encoding="utf-8")

    """
    print(len(train_df))
    print(len(test_df))
    print(len(train_df[train_df["duplicate"] == 0]))
    print(len(train_df[train_df["duplicate"] == 1]))
    print(len(test_df[test_df["duplicate"] == 0]))
    print(len(test_df[test_df["duplicate"] == 1]))
    """

    train_df, test_df = Main.tokenize_and_texts_to_sequences(train_df, test_df)

    """    
    train_dataloader, test_dataloader = Main.get_dataloaders(train_df, test_df)
    for id, (q1, q2, duplicate) in enumerate(test_dataloader):
        print(id)
        print(q1)
        print(type(q1))
        print(q1.size())
    """

    model = Main.run_train(train_df)
    result = Main.run_test(model, test_df)
    print("test result: ", result)
    Main.run_test_2(model)


