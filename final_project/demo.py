from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Merge, Input, Multiply, Concatenate, Lambda
from keras.layers.merge import Multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py
import json
import numpy as np
import os

from keras import backend as K
import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

from IPython.display import Image

from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences

from data_processing import *
from constants import *

##Get VGG Embedding for Random Image
def get_vgg_embed(filename):
    if os.path.exists(filename):
        img = image.load_img(filename, target_size=(224, 224))
    else:
        print("Can't find image at location " + filename)
        return
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    base_model = VGG19(weights='imagenet')
    chopped_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    
    out_embedding = chopped_model.predict(x)
    
    tem = np.sqrt(np.sum(np.multiply(out_embedding, out_embedding), axis=1))
    train_img_data = np.divide(out_embedding, np.transpose(np.tile(tem,(4096,1))))
    
    return train_img_data

def get_ques_vector(question, metadata):
    question_vector = []

    word_index = metadata['ix_to_word']
    for word in word_tokenize(question.lower()):
        if word in word_index:
            question_vector.append(word_index[word])
        else:
            question_vector.append(0)
    question_vector = np.array(pad_sequences([question_vector], maxlen=seq_length))[0]
    question_vector = question_vector.reshape((1,seq_length))
    return question_vector

def test_vqa_model(question_string, img_path, vqa_mod):    
    metadata = get_metadata()

    x = get_vgg_embed(img_path)
    y = get_ques_vector(question_string, metadata)
    pred = vqa_mod.predict([x, y])[0]
    top_pred = pred.argsort()[-5:][::-1]
    labels = [metadata['ix_to_ans'][str(_)] for _ in top_pred]
    print("Question: " + question_string)
    print("Answer: " + labels[0])
    return labels
