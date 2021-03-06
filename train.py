from __future__ import print_function
import numpy as np
from keras.models import model_from_json#load_model
from keras.callbacks import ModelCheckpoint
import os
import argparse
#from models import *
from models_functional import *
from prepare_data import *
from constants import *
import matplotlib.pyplot as plt


def get_model(dropout_rate, model_weights_filename, load_pretrained_weights=False):
    print("Creating Model...")
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())
    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)
    model = vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    if os.path.exists(model_weights_filename) and load_pretrained_weights:
        print("Loading Weights...")
        model.load_weights(model_weights_filename)
    else:
        print("Skipping loading weights.")

    return model

def train(args):
    dropout_rate = 0.5
    train_X, train_y = read_data(args.data_limit)
    val_X, val_y, multi_val_y = get_val_data()
    print("load_weights ", args.load_weights)
    model = get_model(dropout_rate, model_weights_filename, load_pretrained_weights=args.load_weights)
    checkpointer = ModelCheckpoint(filepath=ckpt_model_weights_filename, verbose=1)
    history = model.fit(train_X, train_y, epochs=args.epoch, batch_size=args.batch_size, callbacks=[checkpointer], shuffle="batch", validation_data=(val_X, val_y))
    model.save_weights(model_weights_filename, overwrite=True)
    plot_training_history(history, "model")
    return model, history

def plot_training_history(history, save_filename=None):
    # summarize history for accuracy
    plt.plot(history.history['acc'], marker='o', linestyle='--')
    plt.plot(history.history['val_acc'], marker='o', linestyle='--')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save_filename:
        plt.savefig(str(save_filename) + "_acc.png")
    # summarize history for loss
    plt.close()
    plt.plot(history.history['loss'], marker='o', linestyle='--')
    plt.plot(history.history['val_loss'], marker='o', linestyle='--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save_filename:
        plt.savefig(str(save_filename) + "_loss.png")


def val():
    val_X, val_y, multi_val_y = get_val_data() 
    model = get_model(0.0, model_weights_filename, load_pretrained_weights=True)
    print("Evaluating Accuracy on validation set:")
    metric_vals = model.evaluate(val_X, val_y)
    metrics = zip(model.metrics_names, metric_vals)
    print("")
    for metric_name, metric_val in metrics:
        print(metric_name, " is ", metric_val)

    # Comparing prediction against multiple choice answers
    true_positive = 0
    preds = model.predict(val_X)
    pred_classes = [np.argmax(_) for _ in preds]
    for i, _ in enumerate(pred_classes):
        if _ in multi_val_y[i]:
            true_positive += 1
    true_positive_rate = np.float(true_positive)/len(pred_classes)
    print("True positive rate: ", true_positive_rate)
    return metrics, true_positive_rate


def loop(args):
    for i in range(1, args.num_loops + 1):
        model, _ = train(args)
        if args.save_all:
            model.save_weights(model_weights_filename+"_epoch_"+str(i*args.epoch), overwrite=False)
        metrics, true_positive_rate = val()
        with open("training_log", "a") as val_log:
            val_log.write("After training epoch " + str(args.epoch * i)+"\n")
            for name, value in metrics:
                val_log.write(name + " " + str(value)+"\n")
            val_log.write("True_positive_rate: " + str(true_positive_rate)+"\n")
        print("Finished loop number: ", i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_limit', type=int, default=215359, help='Number of data points to fed for training')
    parser.add_argument('--num_loops', type=int, default=1)
    parser.add_argument('--save_all', type=bool, default=False)
    parser.add_argument('--load_weights', type=bool, default=True)
    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'val':
        val()
    elif args.type == 'loop':
        loop(args)
