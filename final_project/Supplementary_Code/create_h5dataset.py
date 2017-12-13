import os
import pickle
import numpy as np
import json
import h5py
import sys


data_prepro_meta = '/json/data_prepro.json'
meta_data = json.load(open(data_prepro_meta, 'r'))

train_images = meta_data['unique_img_train']
test_images = meta_data['unique_img_test']

train_pkl_dir = "/resnet_pkl"
test_pkl_dir = "/resnet_pkl_test"

pkl_dirs = { "train": "/resnet_pkl", "test": "/resnet_pkl_test"}
f = h5py.File('/output/resnet_imgs.hdf5','w')

for k in pkl_dirs.keys():
	dicts = []

	for filename in os.listdir(pkl_dirs[k]):
	    dicts.append( pickle.load( open( pkl_dirs[k] + "/" + filename, "rb" ) ) )

	full_dictionary = {}

	for individual_dictionary in dicts:
	    for fname in individual_dictionary.keys():
	        full_dictionary[fname] = individual_dictionary[fname]

	num_files = (len(full_dictionary))

	ar = np.zeros( (num_files, 2048) )

	meta_data_dir = "unique_img_" + k
	for count, t in enumerate( meta_data[meta_data_dir]):
	    if t in full_dictionary:
	        ar[count, :] = full_dictionary[t]
	    else:
	        print("File not found! " + t)
	        sys.exit()


	f.create_dataset("images_" + k, data = ar, dtype="<f4")

f.close()





