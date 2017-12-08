from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, \
    Embedding, Merge, Input, Multiply, Concatenate, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop
from compact_bilinear_pooling import CompactBilinearPooling, bili_pooling
import h5py

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating functional text model...")
    input_q = Input((seq_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=True)(input_q)
    x = LSTM(units=512, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=512, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    q_embedding = Dense(1024, activation='tanh')(x)
    return input_q, q_embedding

#
# def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
#     print("Creating functional text model with both layers...")
#     input_q = Input((seq_length,))
#     x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=True)(input_q)
#     lstm1, state_h1, state_c1 = LSTM(units=512, return_sequences=True, return_state=True)(x)
#     lstm2, state_h2, state_c2 = LSTM(units=512, return_sequences=False, return_state=True)(lstm1)
#     concat = Concatenate()([state_h1, state_c1, state_h2, state_c2])
#     q_embedding = Dense(1024, activation='tanh')(concat)
#     return input_q, q_embedding

def img_model(dropout_rate):
    print("Creating functional image model...")
    input_img = Input((4096,))
    img_embedding = Dense(1024, input_dim=4096, activation='tanh')(input_img)
    return input_img, img_embedding

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    input_img, img_embedding = img_model(dropout_rate)
    input_q, q_embedding = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    combined = combine(img_embedding, q_embedding)
    # combined = Dropout(dropout_rate)(combined)
    combined = Dense(1000, activation='tanh')(combined)
    combined = Dropout(dropout_rate)(combined)
    predictions = Dense(num_classes, activation='softmax')(combined)
    fc_model = Model(inputs=[input_img, input_q], outputs=predictions)
    optimizer = RMSprop(lr=0.005)
    fc_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

# Multiply combine
def combine(img_embedding, q_embedding):
    return Multiply()([img_embedding, q_embedding])

# # Multimodal Compact Bilinear Pooling Combine
# def combine(img_embedding, q_embedding):
#     return CompactBilinearPooling(16000)([img_embedding, q_embedding])

# # Bilinear Pooling Combine
# def combine(img_embedding, q_embedding):
#     return Lambda(bili_pooling)([img_embedding, q_embedding])


# # Multimodal Compact Bilinear Pooling Combine
# def combine(img_embedding, q_embedding):
#     return Concatenate(axis=-1)([img_embedding, q_embedding])
#
