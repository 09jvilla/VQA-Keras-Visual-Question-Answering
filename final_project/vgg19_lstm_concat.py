from data_processing import *

def img_model(dropout_rate):
    print("Creating functional image model...")
    input_img = Input((4096,))
    img_embedding = Dense(1024, input_dim=4096, activation='tanh')(input_img)
    return input_img, img_embedding

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating functional text model with both layers...")
    input_q = Input((seq_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_q)
    lstm1, state_h1, state_c1 = LSTM(units=512, return_sequences=True, return_state=True)(x)
    lstm1_d = Dropout(dropout_rate)(lstm1)
    
    lstm2, state_h2, state_c2 = LSTM(units=512, return_sequences=False, return_state=True)(lstm1_d)    
    
    concat = Concatenate()([state_h1, state_c1, state_h2, state_c2])
    concat = Dropout(dropout_rate)(concat)
    
    q_embedding = Dense(1024, activation='tanh')(concat)
    
    return input_q, q_embedding

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    input_img, img_embedding = img_model(dropout_rate)
    input_q, q_embedding = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final functional model...")
    combined = Multiply()([img_embedding, q_embedding])
    combined = Dropout(dropout_rate)(combined)
    combined = Dense(1000, activation='tanh')(combined)
    combined = Dropout(dropout_rate)(combined)
    predictions = Dense(num_classes, activation='softmax')(combined)
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