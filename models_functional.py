from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Merge, Input, Multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating functional text model...")
    input_q = Input((seq_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_q)
    x = LSTM(units=512, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=512, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    q_embedding = Dense(1024, activation='tanh')(x)
    return input_q, q_embedding

def img_model(dropout_rate):
    print("Creating functional image model...")
    input_img = Input((4096,))
    img_embedding = Dense(1024, input_dim=4096, activation='tanh')(input_img)
    return input_img, img_embedding

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    input_img, img_embedding = img_model(dropout_rate)
    input_q, q_embedding = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    combined = Multiply()([img_embedding, q_embedding])
    combined = Dropout(dropout_rate)(combined)
    combined = Dense(1000, activation='tanh')(combined)
    combined = Dropout(dropout_rate)(combined)
    predictions = Dense(num_classes, activation='softmax')(combined)
    fc_model = Model(inputs=[input_img, input_q], outputs=predictions)
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

