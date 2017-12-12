from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, \
    Embedding, Input, Multiply, Concatenate
from keras.optimizers import RMSprop
import keras.backend as K
from compact_bilinear_pooling import CompactBilinearPooling
import tensorflow as tf

# def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
#     print("Creating functional text model...")
#     input_q = Input((seq_length,))
#     x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_q)
#     x = LSTM(units=512, return_sequences=True)(x)
#     x = Dropout(dropout_rate)(x)
#     x = LSTM(units=512, return_sequences=False)(x)
#     x = Dropout(dropout_rate)(x)
#     q_embedding = Dense(1024, activation='tanh')(x)
#     return input_q, q_embedding


def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating functional text model with both layers...")
    input_q = Input((seq_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_q)
    lstm1, state_h1, state_c1 = LSTM(units=512, return_sequences=True, return_state=True)(x)
    lstm1 = Dropout(dropout_rate)(lstm1)
    lstm2, state_h2, state_c2 = LSTM(units=512, return_sequences=False, return_state=True)(lstm1)
    concat = Concatenate()([state_h1, state_c1, state_h2, state_c2])
    concat = Dropout(dropout_rate)(concat)
    q_embedding = Dense(1024, activation='tanh')(concat)
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
    combined = combine(img_embedding, q_embedding)
    combined = Dropout(dropout_rate)(combined)
    combined = Dense(1000, activation='tanh')(combined)
    combined = Dropout(dropout_rate)(combined)
    predictions = Dense(num_classes, activation='softmax')(combined)
    fc_model = Model(inputs=[input_img, input_q], outputs=predictions)
    optimizer = RMSprop(lr=0.001)
    fc_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["acc", vqa_eval_accuracy])
    return fc_model

# # Multiply combine
# def combine(img_embedding, q_embedding):
#     return Multiply()([img_embedding, q_embedding])

# Multimodal Compact Bilinear Pooling Combine
def combine(img_embedding, q_embedding):
    return CompactBilinearPooling(16000)([img_embedding, q_embedding])


def vqa_eval_accuracy(y_true, y_pred):
    trues = K.clip(y_true, min_value=0, max_value=0.3) / 0.3
    preds = K.argmax(y_pred, axis=-1)
    return K.map_fn(gather, [trues, preds], dtype=tf.float32)

def gather(x):
    true, pred = x
    return K.gather(true, pred)