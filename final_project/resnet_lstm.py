from data_processing import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Merge, Input, Multiply, Concatenate, Lambda
from keras.layers.merge import Multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import json
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import os
import argparse
from keras.models import Sequential, Model
import pickle
import logging
from keras import regularizers

from constants import *
from data_processing import *

def img_model(dropout_rate):
    print("Creating functional image model using pre-computed embeddings...")
    input_img = Input(shape=(2048,), name="image_embedding_in")
    img_embedding = Dense(1024, input_dim=2048, kernel_regularizer=regularizers.l2(0.01), activation='tanh', name="image_embedding_fc1_2048")(input_img)
    return input_img, img_embedding

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating functional text model with both layers...")
    input_q = Input(shape=(seq_length,), name="question_in")
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False, name="question_embedding")(input_q)
    lstm1, state_h1, state_c1 = LSTM(units=512, return_sequences=True, return_state=True, name="question_lstm1")(x)
    lstm1 = Dropout(dropout_rate, name="lstm1_dropout")(lstm1)
    lstm2, state_h2, state_c2 = LSTM(units=512, return_sequences=False, return_state=True, name="question_lstm2")(lstm1)    
    
    concat = Concatenate()([state_h1, state_c1, state_h2, state_c2])
    
    q_embedding = Dense(1024, kernel_regularizer=regularizers.l2(0.01), activation='tanh', name="question_fc1")(concat)
    q_embedding = Dropout(dropout_rate, name="q_embedding_dropout")(q_embedding)

    return input_q, q_embedding

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    input_img, img_embedding = img_model(dropout_rate)
    input_q, q_embedding = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final functional model...")
    combined = Multiply(name="img_ques_mul")([img_embedding, q_embedding])
    combined = Dropout(dropout_rate, name="img_ques_dropout1")(combined)
    combined = Dense(1000, activation='tanh', name="img_ques_dense1", kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = Dropout(dropout_rate, name="img_ques_dropout2")(combined)
    predictions = Dense(num_classes, activation='softmax', name="img_ques_dense2")(combined)
    fc_model = Model(inputs=[input_img, input_q], outputs=predictions)
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

def get_model(dropout_rate, model_weights_filename=""):
    print("Creating Model...")
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())

    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)
    model = vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    if model_weights_filename=="":
        print("No model weights file specified. Skipping weight load.")
    elif os.path.exists(model_weights_filename):
        print("Loading Weights...")
        model.load_weights(model_weights_filename)
    else:
        print("Could not find file: " + model_weights_filename + " for loading. Skipping weight load.")

    return model