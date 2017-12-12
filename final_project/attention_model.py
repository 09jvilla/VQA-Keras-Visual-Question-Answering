import tensorflow as tf
from keras import backend as K
from keras.layers import *
from data_processing import *

def Word2VecModel_attn(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating functional text model...")
    input_q = Input((seq_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_q)
    x = LSTM(units=512, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=512, return_sequences=False)(x)
    return input_q, x

def img_model_attn():
    print("Creating functional image model...")
    input_img = Input((49,2048,))
    return input_img

def doConcat(A):
    q_embed = A[0]
    img_embed = A[1]
    q_embed = tf.expand_dims(q_embed, axis=1)
    
    q_embed = tf.tile(q_embed, [1,49,1])
    
    final_tensor = tf.concat([q_embed, img_embed], axis=2)
   
    return final_tensor

def my_matmul(A):
    return tf.matmul(A[0], A[1])

def vqa_w_attn(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    input_q, q_embedding = Word2VecModel_attn(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    input_img = img_model_attn()
    
    concat_regions = Lambda(doConcat)([q_embedding, input_img])
    y_tilde = Dense(units=1024, activation="tanh")(concat_regions)
    g = Dense(units=1024, activation="sigmoid")(concat_regions)
    
    y = Multiply()([y_tilde, g])
    
    alpha = Dense(units=1, use_bias=False, activation="softmax")(y)
    
    
    
    alpha = Permute((2,1))(alpha)
    
    v_hat = Lambda(my_matmul)([alpha, input_img])
    
    v_hat = Dense(units=512, activation="tanh")(v_hat)
    fused = Multiply()([v_hat, q_embedding])
    
    combined = Dense(1000, activation='tanh')(fused)
    combined = Dropout(dropout_rate)(fused)
    predictions = Dense(num_classes, activation='softmax')(combined)
    
    fc_model = Model(inputs=[input_img, input_q], outputs=predictions)
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

def get_model_w_attn(dropout_rate=0.5):
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())

    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)
    model = vqa_w_attn(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    return model
    