from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.utils import plot_model
from keras.models import Model
import json
import pickle
from PIL import Image

base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer("flatten_1").output)
#plot_model(model, show_shapes=True, to_file='model2.png')

##Path to where json file exists with list of jpg images
data_prepro_meta = '/jsonlist/data_prepro.json'
meta_data = json.load(open(data_prepro_meta, 'r'))

feature_dict = {}

##dump data to output folder
pickle_path = '/output/'


img_splits = ['training', 'validation']

for s in img_splits:
	if s == 'training':
		meta_data_key = 'unique_img_train'
	else:
		meta_data_key = 'unique_img_test'

	for count, file_name in enumerate(meta_data[meta_data_key]):

		if (count % 100 == 0):
			print("Num complete for " + s + ": " + str(count))
		
		base_p = "/vqaimgs/" + s + "/"
		img_path = base_p + file_name


		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		features = model.predict(x)

		feature_dict[file_name] = features

		"""
		if (count % 10000 == 0):
			print("Dumping pickle file")
			pickle.dump( feature_dict , open( pickle_path + s + "_" + str(count) + "_feature_dict.p", "wb"))
			feature_dict.clear()
		"""

	print("Completed " + s + ". Dumping last pickle file.")
	pickle.dump( feature_dict , open( pickle_path + s + "_end" + "_feature_dict.p", "wb"))





	
