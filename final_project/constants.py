import os

seq_length =    26
embedding_dim = 300

data_dir = "./data"
output_dir = "./data"
checkpoint_dir = output_dir + "/ckpts"

#Make sure output directory exists
if not os.path.exists(output_dir):
    print("Couldn't find " + output_dir + " so creating it.")
    os.makedirs(output_dir)

#Make sure checkpoint directory exists
if not os.path.exists(checkpoint_dir):
    print("Couldn't find " + checkpoint_dir + " so creating it.")
    os.makedirs(checkpoint_dir)

resnet_train_img = './data_resnet/resnet_train.hdf5'
resnet_test_img = './data_resnet/resnet_test.hdf5'

data_img =                  data_dir + '/data_img.h5'
data_prepo =                data_dir + '/data_prepro.h5'
data_prepo_meta =           data_dir + '/data_prepro.json'
embedding_matrix_filename = checkpoint_dir + '/embeddings_%s.h5'%embedding_dim
glove_path =                data_dir + '/glove.6B.300d.txt'
train_questions_path =      data_dir + '/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json'
train_annotations_path =      data_dir + '/Questions_Train_mscoco/mscoco_val2014_annotations.json'
val_annotations_path =      data_dir + '/validation_data/mscoco_val2014_annotations.json'
ckpt_model_weights_filename =    checkpoint_dir + '/trained_model_weights.h5'
soft_train_y_filename =     data_dir + '/soft_train_y.h5'