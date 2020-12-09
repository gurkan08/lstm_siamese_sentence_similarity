
# keras model
"""
# direct keras (keras API) modules
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, SimpleRNN
from keras.models import Model
"""
# tf==1.15.0 modules
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from params import Params

class LSTMModel(object):

    def __init__(self,
                 max_sentence_size,
                 embed_size,
                 vocab_size,
                 lstm_units,
                 dense_size,
                 label_size):

        q1_input_layer = Input(shape=(max_sentence_size,))
        q2_input_layer = Input(shape=(max_sentence_size,))

        # mask_zero=True zero_padding, trainable=False fasttext embedding init
        if Params.fasttext_embedding_use:
            embed_layer = Embedding(input_dim=vocab_size,
                                    output_dim=embed_size,
                                    mask_zero=True,
                                    weights=[Params.embedding_matrix],
                                    trainable=False) # False
        else:
            embed_layer = Embedding(input_dim=vocab_size,
                                    output_dim=embed_size,
                                    mask_zero=True,
                                    trainable=True) # True

        q1_embed_layer_out = embed_layer(q1_input_layer)
        q2_embed_layer_out = embed_layer(q2_input_layer)

        lstm_layer = LSTM(lstm_units,
                          return_sequences=False,
                          return_state=False,
                          trainable=True)
        # print(lstm_layer.shape)

        q1_lstm_layer_out = lstm_layer(q1_embed_layer_out)
        q2_lstm_layer_out = lstm_layer(q2_embed_layer_out)

        dense_layer = Dense(dense_size,
                            activation="relu",
                            trainable=True)
        # print(dense_layer.shape)

        q1_dense_layer_out = dense_layer(q1_lstm_layer_out)
        q2_dense_layer_out = dense_layer(q2_lstm_layer_out)

        drop_layer = Dropout(rate=Params.drop_out)

        q1_drop_layer_out = drop_layer(q1_dense_layer_out)
        q2_drop_layer_out = drop_layer(q2_dense_layer_out)

        malstm_distance = Lambda(function=lambda x: LSTMModel.exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([q1_drop_layer_out, q2_drop_layer_out])
        #print("--------> ", malstm_distance.shape)

        """
        output_layer = Dense(label_size,
                             activation="softmax",
                             trainable=True)(malstm_distance)
        # print(output_layer.shape)
        """

        self.model = Model([q1_input_layer, q2_input_layer], malstm_distance)

    def get_model(self):
        return self.model

    @staticmethod
    def exponent_neg_manhattan_distance(q1, q2):
        ''' Helper function for the similarity estimate of the LSTMs outputs'''
        return K.exp(-K.sum(K.abs(q1 - q2), axis=1, keepdims=True))

