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
NUM_CLASSES = 2
MODEL_FILE_NAME = "{}_model.h5".format(os.path.splitext(os.path.basename(__file__))[0])

DEBUG_DIR = "__debug__"

def augmentation(img, paramR, paramL):
	height, width, num_channels = img.shape
	
	# crop
	#shift_h = int(width * 0.1)
	#shift_v = int(height * 0.1)
	shift_h = 4
	shift_v = 4
	offset_x = int(random.uniform(0, shift_h * 2))
	offset_y = int(random.uniform(0, shift_v * 2))
	img = tf.image.resize_with_crop_or_pad(img, height + shift_v * 2, width + shift_h * 2)
	img = img[offset_y:offset_y+height, offset_x:offset_x+width,:]
	paramR = (paramR * width + shift_h - offset_x) / width
	paramL = (paramL * width + shift_h - offset_x) / width
	
	paramR = numpy.clip(paramR, a_min = 0, a_max = 1)
	paramL = numpy.clip(paramL, a_min = 0, a_max = 1)
			
	# rotate
	angle = random.uniform(-0.1, 0.1)
	img = scipy.ndimage.rotate(img, angle , axes=(1, 0), reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
	
	return img, paramR, paramL
	

def standardize_img(img):
	mean = numpy.mean(img, axis=None, keepdims=True)
	std = numpy.sqrt(((img - mean)**2).mean(axis=None, keepdims=True))
	return (img - mean) / std


def load_img(file_path):
	img = Image.open(file_path)
	img.load()
	img = numpy.asarray(img, dtype="int32")
	
	# Convert image to grayscale
	r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	img[:,:,0] = gray
	img[:,:,1] = gray
	img[:,:,2] = gray
	
	img = img.astype("float")
	return img


def load_imgs(path_list, ground_params, floor_params, use_augmentation = False, augmentation_factor = 1, use_shuffle = False, debug = False):
    # Calculate number of images
    if use_augmentation:
        num_images = len(path_list) * augmentation_factor
    else:
        num_images = len(path_list)

    X = numpy.zeros((num_images, WIDTH, HEIGHT, 3), dtype=float)
    Y = numpy.zeros((num_images, 2), dtype=float)

    # Load images
    i = 0
    for file_path in path_list:
        file_name = os.path.basename(file_path)
        
        orig_img = load_img(file_path)
        orig_height, orig_width, channels = orig_img.shape
        
        # Crop above shop
        floors = sorted(floor_params[file_name])
        shop = int(floors[len(floors) - 1] * orig_height)
        orig_img = orig_img[shop:orig_height,:,:]
        orig_height, orig_width, channels = orig_img.shape
        
        img = cv2.resize(orig_img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        values = sorted(ground_params[file_name], reverse = True)
        
        width = orig_width
        #Previous loop location 
        valueR = values[0]
        valueL = values[1]
        actual_valueR = valueR * orig_width / width
        actual_valueL = valueL * orig_width / width
        
        if use_augmentation:
            for j in range(augmentation_factor):
                img_tmp, adjusted_valueR, adjusted_valueL = augmentation(img, actual_valueR, actual_valueL)
                
                if debug:
                    output_filename = "{}/{}.png".format(DEBUG_DIR, i)
                    print(output_filename)
                    output_img(img_tmp, adjusted_valueR, adjusted_valueL, output_filename)
                
                X[i,:,:,:] = standardize_img(img_tmp)
                Y[i, 0] = adjusted_valueR
                Y[i, 1] = adjusted_valueL
                i += 1
        else:
            if debug:
                output_filename = "{}/{}.png".format(DEBUG_DIR, i)
                print(output_filename)
                output_img(img, actual_valueR, actual_valueL, output_filename)
                
            X[i,:,:,:] = standardize_img(img)
            Y[i, 0] = actual_valueR
            Y[i, 1] = actual_valueL
            i += 1
        
    if use_shuffle:
        randomize = numpy.arange(len(X))
        numpy.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

    return X, Y

def output_img(img, valueR, valueL, filename):
	img = Image.fromarray(img.astype(numpy.uint8))
	width, height = img.size
	imgdraw = ImageDraw.Draw(img)
	
	imgdraw.line([(width * valueR, 0), (width * valueR, height)], fill = "yellow", width = 3)
	imgdraw.line([(width * valueL, 0), (width * valueL, height)], fill = "yellow", width = 3)
	img.save(filename)
	
	
def output_img2(img, values, filename):
	width, height = img.size
	imgdraw = ImageDraw.Draw(img)
	
	for a in range(0, len(values), 2):
		valueR = values[a]
		valueL = values[a + 1]
		imgdraw.line([(width * valueR, 0), (width * valueR, height)], fill = "yellow", width = 3)
		imgdraw.line([(width * valueL, 0), (width * valueL, height)], fill = "yellow", width = 3)
	img.save(filename)

		
def load_annotation(file_path):
	ground_params = {}
	file = open(file_path, "r")
	while True:
		filename = file.readline().strip()
		if len(filename) == 0: break
		
		columns = file.readline().strip()
		ground_columns = file.readline().strip()
		
		values = []
		data = ground_columns.split(',')
		if len(data) > 0:
			for i in range(len(data)):
				values.append(float(data[i].strip()))
			ground_params[filename] = values
		
	return ground_params

def load_annotation_floor(file_path):
    floor_params = {}
    file = open(file_path, "r")
    while True:
        filename = file.readline().strip()
        if len(filename) == 0: break

        floors = file.readline().strip()

        values = []
        data = floors.split(',')
        if len(data) > 0:
            for i in range(len(data)):
                values.append(float(data[i].strip()))
            floor_params[filename] = values

    return floor_params

def build_model(int_shape, num_params, learning_rate):
	model = tf.keras.Sequential([
		tf.keras.applications.VGG19(input_shape=(WIDTH, HEIGHT, 3), include_top=False, weights='imagenet'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(num_params),
	])
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	
	model.compile(
		loss='mse',
		optimizer=optimizer,
		metrics=['mae', 'mse'])
	
	return model
  

def train(input_dir, model_dir, num_epochs, learning_late, augmentation_factor, output_dir, debug):
    # Load parameters
    ground_params = load_annotation("column_annotation.txt")
    floor_params = load_annotation_floor("floor_annotation.txt")

    # Split the tensor into train and test dataset
    path_list = glob.glob("{}/*.jpg".format(input_dir))
    X, Y = load_imgs(path_list, ground_params, floor_params, use_augmentation = True, augmentation_factor = augmentation_factor, use_shuffle = True, debug = debug)
    #print(X.shape)
    if debug: return

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


def test(input_dir, model_dir, output_dir, debug):
    # Load parameters
    ground_params = load_annotation("column_annotation.txt")
    floor_params = load_annotation_floor("floor_annotation.txt")

    # Split the tensor into train and test dataset
    path_list = glob.glob("{}/*.jpg".format(input_dir))
    X, Y = load_imgs(path_list, ground_params, floor_params, debug = debug)
    if debug: return

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
    accuracy = 0
    accuracy2 = 0
    for i in range(len(path_list)):
        file_name = os.path.basename(path_list[i])
        print(path_list[i])
		
        orig_img = load_img(path_list[i])
        orig_height, orig_width, channels = orig_img.shape

		# Crop sky and shop
        floors = sorted(floor_params[file_name])
        shop = int(floors[len(floors) - 1] * orig_height)
        orig_img = orig_img[shop:orig_height,:,:]
        orig_height, orig_width, channels = orig_img.shape
		
        img = cv2.resize(orig_img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        width = orig_width
        
        Y = []
        # Prediction
        X = numpy.zeros((1, WIDTH, HEIGHT, 3), dtype=float)
        X[0,:,:,:] = standardize_img(img)
        valueR = model.predict(X).flatten()[0]
        R = valueR * width
        valueR = numpy.clip(valueR * width / orig_width, a_min = 0, a_max = 1)
        valueL = model.predict(X).flatten()[1]
        L = valueL * width
        valueL = numpy.clip(valueL * width / orig_width, a_min = 0, a_max = 1)
        if valueL < 0.05: break
        Y.append(valueR)
        Y.append(valueL)
        w = 0;
        correct = 0;
        w2 = 0;
        correct2 = 0;
        R_truth = ground_params[file_name][1] * width
        L_truth = ground_params[file_name][0] * width
        correct3 = 0
        correct4 = 0
        for j in range(width):
            if (j < L or j > R):
                w += 1;
                if (j < L_truth or j > R_truth):
                    correct += 1
            
            else:
                w2 += 1;
                if (j > L_truth and j < R_truth):
                    correct2 += 1
                    
            if (j < L_truth or j > R_truth):
                    correct3 += 1
            
            if (j > L_truth and j < R_truth):
                    correct4 += 1
        
        #print((1 - floors[len(floors) - 1]) * correct3 / width)
        #print((1 - floors[len(floors) - 1]) * correct4 / width)
        
        accuracy += (correct / w)
        if (w2 != 0):
            accuracy2 += (correct2 / w2)
        
        
        # Save prediction image
        file_name = "{}/{}".format(output_dir, os.path.basename(path_list[i]))
        output_img2(Image.open(path_list[i]), Y, file_name)
    
    #print(accuracy / len(path_list))
    #print(accuracy2 / len(path_list))


def main():	
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=["train", "test"])
    parser.add_argument('--input_dir', required=True, help="path to folder containing images")
    parser.add_argument('--output_dir', default="out", help="where to put output files")
    parser.add_argument('--model_dir', default="models", help="path to folder containing models")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--augmentation_factor', type=int, default=100)
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
        else:
            files = glob.glob("{}/*".format(DEBUG_DIR))
            for f in files:
                os.remove(f)

    if args.mode == "train":
        train(args.input_dir, args.model_dir, args.num_epochs, args.learning_rate, args.augmentation_factor, args.output_dir, args.debug)
    elif args.mode == "test":
        test(args.input_dir, args.model_dir, args.output_dir, args.debug)
    else:
        print("Invalid mode is specified {}".format(args.mode))
        exit(1)
	

if __name__== "__main__":
	main()
