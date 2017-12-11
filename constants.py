
seq_length =    26
embedding_dim = 300

data_img =                  'data/data_img.h5'
data_prepo =                'data/data_prepro.h5'
data_prepo_meta =           'data/data_prepro.json'
embedding_matrix_filename = 'data/ckpts/embeddings_%s.h5'%embedding_dim
glove_path =                'data/glove.6B.300d.txt'
train_questions_path =      'data/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json'
val_annotations_path =      'data/validation_data/mscoco_val2014_annotations.json'
train_annotations_path =    'data/mscoco_train2014_annotations.json'

model_name = "Test"
ckpt_model_weights_filename =    'data/ckpts/model_' + model_name + '_weights.{epoch:02d}.h5'
model_weights_filename =    'data/model_' + model_name + '_weights.h5'
history_filename = 'model_' + model_name + '_history.csv'
