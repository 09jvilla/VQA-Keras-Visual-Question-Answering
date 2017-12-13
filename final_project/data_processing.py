from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Merge, Input, Multiply, Concatenate, Lambda
from keras.layers.merge import Multiply
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py
import os

import numpy as np
from keras.utils.np_utils import to_categorical
import json

from constants import *

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
    return v

def read_data(data_limit, data_img=data_img, embedding_img_dim=4096):
    print("Reading Data...")
    img_data = h5py.File(data_img)
    ques_data = h5py.File(data_prepo)
  
    img_data = np.array(img_data['images_train'])
    img_pos_train = ques_data['img_pos_train'][:data_limit]
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    
    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(embedding_img_dim,1))))

    #shifting padding to left side
    ques_train = np.array(ques_data['ques_train'])[:data_limit, :]
    ques_length_train = np.array(ques_data['ques_length_train'])[:data_limit]
    ques_train = right_align(ques_train, ques_length_train)

    train_X = [train_img_data, ques_train]
    # NOTE should've consturcted one-hots using exhausitve list of answers, cause some answers may not be in dataset
    # To temporarily rectify this, all those answer indices is set to 1 in validation set
    train_y = to_categorical(ques_data['answers'])[:data_limit, :]

    return train_X, train_y

def get_val_data(data_img=data_img, embedding_img_dim=4096):
    img_data = h5py.File(data_img)
    ques_data = h5py.File(data_prepo)
    metadata = get_metadata()
    with open(val_annotations_path, 'r') as an_file:
        annotations = json.loads(an_file.read())

    img_data = np.array(img_data['images_test'])
    img_pos_train = ques_data['img_pos_test']
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(embedding_img_dim,1))))

    ques_train = np.array(ques_data['ques_test'])
    ques_length_train = np.array(ques_data['ques_length_test'])
    ques_train = right_align(ques_train, ques_length_train)

    # Convert all last index to 0, coz embeddings were made that way :/
    for _ in ques_train:
        if 12602 in _:
            _[_==12602] = 0

    val_X = [train_img_data, ques_train]

    ans_to_ix = {str(ans):int(i) for i,ans in metadata['ix_to_ans'].items()}
    ques_annotations = {}
    for _ in annotations['annotations']:
        idx = ans_to_ix.get(_['multiple_choice_answer'].lower())
        _['multiple_choice_answer_idx'] = 1 if idx in [None, 1000] else idx
        ques_annotations[_['question_id']] = _

    abs_val_y = [ques_annotations[ques_id]['multiple_choice_answer_idx'] for ques_id in ques_data['question_id_test']]
    abs_val_y = to_categorical(np.array(abs_val_y))

    multi_val_y = [list(set([ans_to_ix.get(_['answer'].lower()) for _ in ques_annotations[ques_id]['answers']])) for ques_id in ques_data['question_id_test']]
    for i,_ in enumerate(multi_val_y):
        multi_val_y[i] = [1 if ans in [None, 1000] else ans for ans in _]

    return val_X, abs_val_y, multi_val_y


def get_metadata():
    meta_data = json.load(open(data_prepo_meta, 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data

def prepare_embeddings(num_words, embedding_dim, metadata):
    if os.path.exists(embedding_matrix_filename):
        with h5py.File(embedding_matrix_filename) as f:
            print("Found matrix so no need to recompute!")
            return np.array(f['embedding_matrix'])

    print("Embedding Data...")
    with open(train_questions_path, 'r') as qs_file:
        questions = json.loads(qs_file.read())
        texts = [str(_['question']) for _ in questions['questions']]
    
    embeddings_index = {}
    with open(glove_path, 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((num_words, embedding_dim))
    word_index = metadata['ix_to_word']

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
   
    with h5py.File(embedding_matrix_filename, 'w') as f:
        f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix


def get_soft_train_y(data_limit):
    '''
    Get Y labels for training data.
    Each label is a weighted vector of the
    fraction of time that an answer of that
    id appeared in the ground truth
    '''
    if os.path.exists(soft_train_y_filename):
        with h5py.File(soft_train_y_filename) as f:
            return np.array(f['soft_train_y'])

    ques_data = h5py.File(data_prepo)
    metadata = get_metadata()
    with open(train_annotations_path, 'r') as an_file:
        annotations = json.loads(an_file.read())

    ans_to_ix = {str(ans):int(i) for i,ans in metadata['ix_to_ans'].items()}
    ques_annotations = {}
    for a in annotations['annotations']:
        ques_annotations[a['question_id']] = a

    m = min(len(ques_data['question_id_train']), data_limit)
    num_classes = 1000
    soft_train_y = np.zeros((m, num_classes))

    for i, ques_id in enumerate(ques_data['question_id_train']):
        if i < m:
            num_ans = len(ques_annotations[ques_id]['answers'])
            for ans in ques_annotations[ques_id]['answers']:
                ix = ans_to_ix.get(ans['answer'].lower())
                ix = 1 if ix in [None, 1000] else ix
                soft_train_y[i, ix] += 1./num_ans
    with h5py.File(soft_train_y_filename, 'w') as f:
        f.create_dataset('soft_train_y', data=soft_train_y)

    return soft_train_y

def get_soft_val_y():
    '''
    Get Y labels for validation data.
    Each label is a weighted vector of the
    fraction of time that an answer of that
    id appeared in the ground truth
    '''
    ques_data = h5py.File(data_prepo)
    metadata = get_metadata()
    with open(val_annotations_path, 'r') as an_file:
        annotations = json.loads(an_file.read())

    ans_to_ix = {str(ans):int(i) for i,ans in metadata['ix_to_ans'].items()}
    ques_annotations = {}
    for a in annotations['annotations']:
        ques_annotations[a['question_id']] = a

    m = len(ques_data['question_id_test'])
    num_classes = 1000
    soft_val_y = np.zeros((m, num_classes))

    for i, ques_id in enumerate(ques_data['question_id_test']):
        if i < m:
            num_ans = len(ques_annotations[ques_id]['answers'])
            for ans in ques_annotations[ques_id]['answers']:
                ix = ans_to_ix.get(ans['answer'].lower())
                ix = 1 if ix in [None, 1000] else ix
                soft_val_y[i, ix] += 1./num_ans

    return soft_val_y