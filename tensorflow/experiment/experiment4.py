import datetime
import os
from PIL import Image, ImageDraw
import numpy
import scipy.misc
import glob
import scipy
import random
import argparse
import cv2

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


HEIGHT = 160
WIDTH = 160
NUM_CHANNELS = 3
NUM_CLASSES = 1
MODEL_FILE_NAME = "experiment4_model.h5"

DEBUG_DIR = "__debug__"

def augmentation(x, y):
	height, width, num_channels = x.shape
	
	# crop
	#shift_h = int(width * 0.1)
	#shift_v = int(height * 0.1)
	shift_h = 4
	shift_v = 4
	offset_x = int(random.uniform(0, shift_h * 2))
	offset_y = int(random.uniform(0, shift_v * 2))
	x = tf.image.resize_with_crop_or_pad(x, height + shift_v * 2, width + shift_h * 2)
	x = x[offset_y:offset_y+height, offset_x:offset_x+width,:]
	y = (y * height + shift_v - offset_y) / height
	if y < 0 or y > 1:
		y = 0
	
	# flip
	x = tf.image.random_flip_left_right(x)
		
	# rotate
	angle = random.uniform(-0.5, 0.5)
	x = scipy.ndimage.rotate(x, angle , axes=(1, 0), reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
	
	return x, y
	

def standardize_img(x):
	mean = numpy.mean(x, axis=None, keepdims=True)
	std = numpy.sqrt(((x - mean)**2).mean(axis=None, keepdims=True))
	return (x - mean) / std


def load_img(file_path):
	img = Image.open(file_path)
	img.load()
	img = numpy.asarray(img, dtype="int32")
	img = img.astype("float")
	return img


def load_imgs(path_list, params, use_augmentation = False, augmentation_factor = 1, use_shuffle = False, all_floors = False, debug = False):
	# Calculate number of images
	num_images = 0
	for file_path in path_list:
		file_name = os.path.basename(file_path)
		if use_augmentation:
			if all_floors:
				num_images += (len(params[file_name]) + 1) * augmentation_factor
			else:
				num_images += augmentation_factor
		else:
			if all_floors:
				num_images += len(params[file_name]) + 1
			else:
				num_images += 1

	X = numpy.zeros((num_images, WIDTH, HEIGHT, 3), dtype=float)
	Y = numpy.zeros((num_images), dtype=float)
	
	# Load images
	i = 0
	for file_path in path_list:	
		orig_img = load_img(file_path)
		orig_height = orig_img.shape[0]
		imgx = cv2.resize(orig_img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
		file_name = os.path.basename(file_path)
		file_base, file_ext = os.path.splitext(file_path)
		
		values = sorted(params[file_name], reverse = True)
		values.append(0.0)

		height = orig_height
		for y in values:
			actual_y = y * orig_height / height
			
			if use_augmentation:
				for j in range(augmentation_factor):
					img_tmp, adjusted_y = augmentation(imgx, actual_y)
					
					if debug:
						output_filename = "{}/{}.png".format(DEBUG_DIR, i)
						print(output_filename)
						output_img(img_tmp, adjusted_y, output_filename)
										
					X[i,:,:,:] = standardize_img(img_tmp)
					Y[i] = adjusted_y
					i += 1					
			else:
				X[i,:,:,:] = standardize_img(imgx)
				Y[i] = actual_y
				i += 1

			if not all_floors: break
			
			# Update image
			if y > 0:
				height = int(orig_height * y)
				imgx = orig_img[0:height,:,:]
				imgx = cv2.resize(imgx, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
			
	if use_shuffle:
		randomize = numpy.arange(len(X))
		numpy.random.shuffle(randomize)
		X = X[randomize]
		Y = Y[randomize]

	return X, Y

def output_img(x, y, filename):
	print(x.shape)
	img = Image.fromarray(x.astype(numpy.uint8))
	w, h = img.size
	imgdraw = ImageDraw.Draw(img)
	
	imgdraw.line([(0, h * y), (w, h * y)], fill = "yellow", width = 3)
	img.save("{}".format(filename))		
		
def load_annotation(file_path):
	params = {}
	file = open(file_path, "r")
	for line in file.readlines():
		line = line.strip()
		values = []
		data = line.split(',')
		if len(data) > 1:
			for i in range(1,len(data)):
				values.append(float(data[i].strip()))
			params[data[0]] = values
		
	return params


def build_model(int_shape, num_params, learning_rate):
	model = tf.keras.Sequential([
		tf.keras.applications.MobileNetV2(input_shape=(WIDTH, HEIGHT, 3), include_top=False, weights='imagenet'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(num_params),
	])
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	
	model.compile(
		loss='mse',
		optimizer=optimizer,
		metrics=['mae', 'mse'])
	
	return model
  

def train(input_dir, model_dir, num_epochs, learning_late, augmentation_factor, all_floors, output_dir, debug):
	# Load parameters
	params = load_annotation("facade_annotation.txt")

	# Split the tensor into train and test dataset
	path_list = glob.glob("{}/*.jpg".format(input_dir))
	X, Y = load_imgs(path_list, params, use_augmentation = True, augmentation_factor = augmentation_factor, use_shuffle = True, all_floors = all_floors, debug = debug)
	print(X.shape)
	
	# Build model
	model = build_model((HEIGHT, WIDTH, NUM_CHANNELS), NUM_CLASSES, learning_late)

	# Setup for Tensorboard
	log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	file_writer = tf.summary.create_file_writer(log_dir + "\\metrics")
	file_writer.set_as_default()
	tensorboard_callback = TensorBoard(
		log_dir=log_dir,
		update_freq='batch',
		histogram_freq=1)

	# Training model
	model.fit(X, Y,
		epochs=num_epochs,
		validation_split = 0.2,
		callbacks=[tensorboard_callback])

	# Save the model
	model.save("{}/{}".format(model_dir, MODEL_FILE_NAME))


def test(input_dir, model_dir, all_floors, output_dir):
	# Load parameters
	params = load_annotation("facade_annotation.txt")

	# Split the tensor into train and test dataset
	path_list = glob.glob("{}/*.jpg".format(input_dir))
	X, Y = load_imgs(path_list, params, all_floors = all_floors)
		  
	# Load the model
	model = tf.keras.models.load_model("{}/{}".format(model_dir, MODEL_FILE_NAME))
		
	# Evaluation
	model.evaluate(X, Y)
	
	# Prediction
	predictedY = model.predict(X).flatten()

	# Write the prediction to a file
	file = open("{}/prediction.txt".format(output_dir), "w")
	for i in range(len(path_list)):
		file_name = os.path.basename(path_list[i])
		file.write("{},{}\n".format(file_name, predictedY[i]))
	file.close()

	# Save the predicted images
	for i in range(len(path_list)):				
		print(path_list[i])
		orig_x = load_img(path_list[i])
		orig_height = orig_x.shape[0]

		x = cv2.resize(orig_x, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
		height = orig_height
		
		# Repeatedly predict floors
		Y = []
		while True:		
			# Prediction
			X = numpy.zeros((1, WIDTH, HEIGHT, 3), dtype=float)
			X[0,:,:,:] = standardize_img(x)
			y = model.predict(X).flatten()[0]
			y = numpy.clip(y * height / orig_height, a_min = 0, a_max = 1)
			if height * y < 20: break
			Y.append(y)
			
			if not all_floors: break
			
			# Update image
			height = int(orig_height * y)
			x = orig_x[0:height,:,:]
			x = cv2.resize(x, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
		
		# Load image
		file_name = os.path.basename(path_list[i])
		img = Image.open(path_list[i])
		w, h = img.size
		imgdraw = ImageDraw.Draw(img)
		
		for y in Y:
			imgdraw.line([(0, h * y), (w, h * y)], fill = "yellow", width = 3)
		img.save("{}/{}".format(output_dir, file_name))


def main():	
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', required=True, choices=["train", "test"])
	parser.add_argument('--input_dir', required=True, help="path to folder containing images")
	parser.add_argument('--output_dir', default="out", help="where to put output files")
	parser.add_argument('--model_dir', default="models", help="path to folder containing models")
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	parser.add_argument('--augmentation_factor', type=int, default=100)
	parser.add_argument('--all_floors', action="store_true", help="Use all floors")
	parser.add_argument('--debug', action="store_true", help="Output debug information")
	args = parser.parse_args()	

	# Create output directory
	if not os.path.isdir(args.output_dir):
		os.mkdir(args.output_dir)

	# Create model directory
	if not os.path.isdir(args.model_dir):
		os.mkdir(args.model_dir)

	# Create debug directory
	if args.debug:
		if not os.path.isdir(DEBUG_DIR):
			os.mkdir(DEBUG_DIR)

	if args.mode == "train":
		train(args.input_dir, args.model_dir, args.num_epochs, args.learning_rate, args.augmentation_factor, args.all_floors, args.output_dir, args.debug)
	elif args.mode == "test":
		test(args.input_dir, args.model_dir, args.all_floors, args.output_dir)
	else:
		print("Invalid mode is specified {}".format(args.mode))
		exit(1)
	

if __name__== "__main__":
	main()
