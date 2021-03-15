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

DEBUG_DIR = "__debug__"

def standardize_img(x):
    mean = numpy.mean(x, axis=None, keepdims=True)
    std = numpy.sqrt(((x - mean)**2).mean(axis=None, keepdims=True))
    return (x - mean) / std

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

def load_img_debug(file_path):
    img = Image.open(file_path)
    img.load()
    img = numpy.asarray(img, dtype="int32")
    
    return img
    
def load_img_balc(file_path):
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

def load_imgs(path_list, params, use_augmentation = False, augmentation_factor = 1, use_shuffle = False, all_floors = False, debug = False):
    # Calculate number of images
    num_images = 0
    for file_path in path_list:
        file_name = os.path.basename(file_path)
        if use_augmentation:
            if all_floors:
                num_images += (int(len(params[file_name]) / 2) + 1) * augmentation_factor
            else:
                num_images += augmentation_factor
        else:
            if all_floors:
                num_images += int(len(params[file_name]) / 2) + 1
            else:
                num_images += 1

    X = numpy.zeros((num_images, WIDTH, HEIGHT, 3), dtype=float)
    Y = numpy.zeros((num_images, 2), dtype=float)

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
        values.append(0.0)
        values.append(0.0)
        values.append(0.0)

        height = orig_height
        for a in range(0, len(values) - 1, 4):
            actual_floor = values[a] * orig_height / height
            actual_Lbal = values[a + 1] * orig_height / height
            actual_Sbal = values[a + 2] * orig_height / height
            actual_window = values[a + 3] * orig_height / height
            
            if use_augmentation:
                for j in range(augmentation_factor):
                    img_tmp, adjusted_floor, adjusted_Lbal, adjusted_Sbal, adjusted_window = augmentation(imgx, actual_floor, actual_Lbal, actual_Sbal, actual_window)
                    
                    if debug:
                        output_filename = "{}/{}.png".format(DEBUG_DIR, i)
                        print(output_filename)
                        output_img(img_tmp, adjusted_floor, adjusted_Lbal, adjusted_Sbal, adjusted_window, output_filename)
                                        
                    X[i,:,:,:] = standardize_img(img_tmp)
                    Y[i, 0] = adjusted_Lbal
                    Y[i, 1] = adjusted_Sbal
                    i += 1
            else:
                X[i,:,:,:] = standardize_img(imgx)
                Y[i, 0] = actual_Lbal
                Y[i, 1] = actual_Sbal
                i += 1

            if not all_floors: break
            
            # Update image
            if values[a] > 0:
                height = int(orig_height * values[a + 3])
                imgx = orig_img[0:height,:,:]
                imgx = cv2.resize(imgx, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
    if use_shuffle:
        randomize = numpy.arange(len(X))
        numpy.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

    return X, Y

def load_imgs_horiz(path_list, params, use_augmentation = False, augmentation_factor = 1, use_shuffle = False, all_floors = False, debug = False):
    # Calculate number of images
    num_images = 0
    for file_path in path_list:
        file_name = os.path.basename(file_path)
        if use_augmentation:
            if all_floors:
                num_images += (int(len(params[file_name]) / 2) + 1) * augmentation_factor
            else:
                num_images += augmentation_factor
        else:
            if all_floors:
                num_images += int(len(params[file_name]) / 2) + 1
            else:
                num_images += 1

    X = numpy.zeros((num_images, WIDTH, HEIGHT, 3), dtype=float)
    Y = numpy.zeros((num_images, 2), dtype=float)

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
        values.append(0.0)

        height = orig_height
        for a in range(0, len(values), 2):
            actual_bot = values[a] * orig_height / height
            actual_top = values[a + 1] * orig_height / height
            
            if use_augmentation:
                for j in range(augmentation_factor):
                    img_tmp, adjusted_bot, adjusted_top = augmentation(imgx, actual_bot, actual_top)
                    
                    if debug:
                        output_filename = "{}/{}.png".format(DEBUG_DIR, i)
                        print(output_filename)
                        output_img(img_tmp, adjusted_bot, adjusted_top, output_filename)
                                        
                    X[i,:,:,:] = standardize_img(img_tmp)
                    Y[i, 0] = adjusted_bot
                    Y[i, 1] = adjusted_top
                    i += 1					
            else:
                X[i,:,:,:] = standardize_img(imgx)
                Y[i, 0] = actual_bot
                Y[i, 1] = actual_top
                i += 1

            if not all_floors: break
            
            # Update image
            if values[a] > 0:
                height = int(orig_height * values[a + 1])
                imgx = orig_img[0:height,:,:]
                imgx = cv2.resize(imgx, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
    if use_shuffle:
        randomize = numpy.arange(len(X))
        numpy.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

    return X, Y
    
def load_imgs_vert(path_list, column_params, floor_params, use_augmentation = False, augmentation_factor = 1, use_shuffle = False, all_columns = False, debug = False):
    # Calculate number of images
    num_images = 0
    for file_path in path_list:
        file_name = os.path.basename(file_path)
        if use_augmentation:
            if all_columns:
                num_images += (int(len(column_params[file_name]) / 2) + 1) * augmentation_factor
            else:
                num_images += augmentation_factor
        else:
            if all_columns:
                num_images += int(len(column_params[file_name]) / 2) + 1
            else:
                num_images += 1

    X = numpy.zeros((num_images, WIDTH, HEIGHT, 3), dtype=float)
    Y = numpy.zeros((num_images, 2), dtype=float)

    # Load images
    i = 0
    for file_path in path_list:
        file_name = os.path.basename(file_path)
        
        orig_img = load_img(file_path)
        orig_height, orig_width, channels = orig_img.shape
        
        # Crop sky and shop
        floors = sorted(floor_params[file_name])
        roof = int(floors[0] * orig_height)
        shop = int(floors[len(floors) - 1] * orig_height)
        orig_img = orig_img[roof:shop,:,:]
        orig_height, orig_width, channels = orig_img.shape
        
        img = cv2.resize(orig_img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        values = sorted(column_params[file_name], reverse = True)
        values.append(0.0)
        values.append(0.0)
        
        width = orig_width
        for a in range(0, len(values), 2):
            valueR = values[a]
            valueL = values[a + 1]
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

            if not all_columns: break
            
            # Update image
            if valueL > 0:
                width = int(orig_width * valueL)
                img = orig_img[:,0:width,:]
                img = cv2.resize(img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
    if use_shuffle:
        randomize = numpy.arange(len(X))
        numpy.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

    return X, Y
    
def load_annotation_horiz(file_path):
    floor_params = {}
    file = open(file_path, "r")
    while True:
        filename = file.readline().strip()
        if len(filename) == 0: break
        
        floors = file.readline().strip()
        
        values = []
        data = floors.split(',')
        if len(data) > 0:
            for i in range(1, len(data)):
                values.append(float(data[i].strip()))
            floor_params[filename] = values
            
    return floor_params
    
def load_annotation(file_path):
	column_params = {}
	file = open(file_path, "r")
	while True:
		filename = file.readline().strip()
		if len(filename) == 0: break
		
		columns = file.readline().strip()
		ground_columns = file.readline().strip()
		
		values = []
		data = columns.split(',')
		if len(data) > 0:
			for i in range(len(data)):
				values.append(float(data[i].strip()))
			column_params[filename] = values
		
	return column_params
    
def load_annotation_balc(file_path):
    floor_params = {}
    file = open(file_path, "r")
    while True:
        filename = file.readline().strip()
        if len(filename) == 0: break
        
        floors = file.readline().strip()
        
        values = []
        data = floors.split(',')
        if len(data) > 0:
            for i in range(1, len(data)):
                values.append(float(data[i].strip()))
            floor_params[filename] = values
            
    return floor_params

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

def test():
    all_floors = True
    all_columns = True
    debug = False
    
    # FLOOR
    # Load parameters
    params = load_annotation_horiz("floor_annotation.txt")

    # Split the tensor into train and test dataset
    path_list = glob.glob("../../ECP/image_test/*.jpg")
    X, Y = load_imgs_horiz(path_list, params, all_floors = all_floors)
          
    # Load the model
    model = tf.keras.models.load_model("../all_floors_experiment/models/all_floors_experiment2_model.h5")

    # Save the predicted images
    Y_horizontal = []
    for i in range(len(path_list)):				
        #print(path_list[i])
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
            y_bot = model.predict(X).flatten()[0]
            y_bot = numpy.clip(y_bot * height / orig_height, a_min = 0, a_max = 1)
            y_top = model.predict(X).flatten()[1]
            y_top = numpy.clip(y_top * height / orig_height, a_min = 0, a_max = 1)
            if y_bot < 0.05: break
            if y_top < 0.05: break
            Y.append(y_bot * orig_height)
            Y.append(y_top * orig_height)
            
            if not all_floors: break
            
            # Update image
            height = int(orig_height * y_top)
            x = orig_x[0:height,:,:]
            x = cv2.resize(x, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
        Y_horizontal.append(Y)

    # COLUMN SECTION
    # Load parameters
    column_params = load_annotation("column_annotation.txt")
    floor_params = load_annotation_floor("floor_annotation_copy.txt")

    # Split the tensor into train and test dataset
    path_list = glob.glob("../../ECP/image_test/*.jpg")
    X, Y = load_imgs_vert(path_list, column_params, floor_params, all_columns = all_columns, debug = debug)
    
    # Load the model
    model = tf.keras.models.load_model("../columns_experiment/models/columns_experiment4_model.h5")

    # Save the predicted images
    Y_vertical = []
    for i in range(len(path_list)):
        file_name = os.path.basename(path_list[i])
        #print(path_list[i])
		
        orig_img = load_img(path_list[i])
        orig_height, orig_width, channels = orig_img.shape

		# Crop sky and shop
        floors = sorted(floor_params[file_name])
        roof = int(floors[0] * orig_height)
        shop = int(floors[len(floors) - 1] * orig_height)
        orig_img = orig_img[roof:shop,:,:]
        orig_height, orig_width, channels = orig_img.shape
		
        img = cv2.resize(orig_img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        width = orig_width
        
        # Repeatedly predict columns
        Y = []
        while True:		
            # Prediction
            X = numpy.zeros((1, WIDTH, HEIGHT, 3), dtype=float)
            X[0,:,:,:] = standardize_img(img)
            valueR = model.predict(X).flatten()[0]
            valueR = numpy.clip(valueR * width / orig_width, a_min = 0, a_max = 1)
            valueL = model.predict(X).flatten()[1]
            valueL = numpy.clip(valueL * width / orig_width, a_min = 0, a_max = 1)
            if valueL < 0.05: break
            Y.append(valueR * orig_width)
            Y.append(valueL * orig_width)
            
            if not all_columns: break
            
            # Update image
            width = int(orig_width * valueL)
            img = orig_img[:,0:width,:]
            img = cv2.resize(img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        Y_vertical.append(Y)
    
    for Y in path_list:
        orig_img = load_img(Y)
        orig_height, orig_width, channels = orig_img.shape
        img_tmp = os.path.split(Y)[1]
        for i in range(len(floor_params[img_tmp])):
            params[img_tmp][i] *= orig_height
        
        for i in range(len(column_params[img_tmp])):
            column_params[img_tmp][i] *= orig_width
    
    # Load parameters
    params3 = load_annotation_balc("balcony_annotation.txt")

    # Split the tensor into train and test dataset
    #path_list = glob.glob("{}/*.jpg".format(input_dir))
    X, Y = load_imgs(path_list, params3, all_floors = True)
    
    # Load the model
    model = tf.keras.models.load_model("../balcony_experiment/models/balcony_experiment2_model.h5")
        
    # Evaluation
    model.evaluate(X, Y)

    # Prediction
    predictedY = model.predict(X).flatten()

    # Write the prediction to a file
    file = open("out/prediction.txt", "w")
    for i in range(len(path_list)):
        file_name = os.path.basename(path_list[i])
        file.write("{},{}\n".format(file_name, predictedY[i]))
    file.close()
    
    # Save the predicted images
    Y = []
    for i in range(len(path_list)):
        img_tmp = os.path.split(path_list[i])[1]
        #print(path_list[i])
        orig_x = load_img_balc(path_list[i])
        orig_height = orig_x.shape[0]

        x = cv2.resize(orig_x, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        height = orig_height
        
        # Repeatedly predict floors
        why = []
        a = 0
        while True:		
            # Prediction
            X = numpy.zeros((1, WIDTH, HEIGHT, 3), dtype=float)
            X[0,:,:,:] = standardize_img(x)
            y_floor = params3[img_tmp][len(params3[img_tmp]) - 4 * a - 1]
            y_Lbal = model.predict(X).flatten()[0]
            y_Lbal = numpy.clip(y_Lbal * height / orig_height, a_min = 0, a_max = 1)
            y_Sbal = model.predict(X).flatten()[1]
            y_Sbal = numpy.clip(y_Sbal * height / orig_height, a_min = 0, a_max = 1)
            y_window = params3[img_tmp][len(params3[img_tmp]) - 4 * a - 4]
            if y_floor < 0.05: break
            if y_Lbal < 0.05: break
            if y_Sbal < 0.05: break
            if y_window < 0.05: break
            why.append(y_floor)
            why.append(y_Lbal)
            why.append(y_Sbal)
            why.append(y_window)
            
            if not all_floors: break
            
            # Update image
            height = int(orig_height * y_window)
            x = orig_x[0:height,:,:]
            x = cv2.resize(x, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
            a = a + 1
            
        #print(why)
        Y.append(why)



    Y_horizontal_truth = params
    Y_vertical_truth = column_params
    #Y
    
    img_num = 0
    total_err = 0
    for image in path_list:
        img_tmp = os.path.split(image)[1]
        img = load_img_debug(image)
        facade_error = 0
        orig_img = load_img(image)
        orig_height, orig_width, channels = orig_img.shape
        print(image)
        tot_union = 0
        for a in range(0, len(Y_horizontal[img_num]), 2):
            if (a >= len(Y_horizontal[img_num])):
                break
                
            if (a >= len(Y_horizontal_truth[img_tmp])):
                break
            
            #window_bot = Y_horizontal[img_num][a]
            window_top = Y_horizontal[img_num][a + 1]
            window_bot_truth = params3[img_tmp][len(params3[img_tmp]) - 2 * a - 3] * orig_height
            window_top_truth = Y_horizontal_truth[img_tmp][a + 1]
            Sbal_top = Y[img_num][2 * a + 2] * orig_height
            Lbal_top = Y[img_num][2 * a + 1] * orig_height
            window_bot = Sbal_top
            if (Sbal_top < window_top):
                window_bot = window_top
            
            if (Sbal_top < window_top):
                Sbal_top = window_top
            
            if (Sbal_top > window_bot):
                Sbal_top = window_bot
                
            
            if (Sbal_top > Lbal_top):
                Lbal_top = Sbal_top
                
            if (Lbal_top > window_bot):
                Lbal_top = window_bot
            
            
            
            for b in range(0, len(Y_vertical[img_num]), 2):
                if (b >= len(Y_vertical[img_num])):
                    break
                    
                if (b >= len(Y_vertical_truth[img_tmp])):
                    break
                
                window_right = Y_vertical[img_num][b]
                window_left = Y_vertical[img_num][b + 1]
                window_right_truth = Y_vertical_truth[img_tmp][len(Y_vertical_truth[img_tmp]) - b - 1]
                window_left_truth = Y_vertical_truth[img_tmp][len(Y_vertical_truth[img_tmp]) - b - 2]
                
                
                # Output predicted window
                img[int(window_top):int(Sbal_top), int(window_left):int(window_right), 2] = numpy.ones((int(Sbal_top) - int(window_top), int(window_right) - int(window_left))) * 255
                img[int(window_top):int(Sbal_top), int(window_left):int(window_right), 1] = img[int(window_top):int(Sbal_top), int(window_left):int(window_right), 1] / 2
                img[int(window_top):int(Sbal_top), int(window_left):int(window_right), 0] = img[int(window_top):int(Sbal_top), int(window_left):int(window_right), 0] / 2
                
                # Output predicted small balcony
                img[int(Sbal_top):int(Lbal_top), int(window_left):int(window_right), 1] = numpy.ones((int(Lbal_top) - int(Sbal_top), int(window_right) - int(window_left))) * 255
                img[int(Sbal_top):int(Lbal_top), int(window_left):int(window_right), 2] = img[int(Sbal_top):int(Lbal_top), int(window_left):int(window_right), 2] / 4 * 3
                img[int(Sbal_top):int(Lbal_top), int(window_left):int(window_right), 0] = img[int(Sbal_top):int(Lbal_top), int(window_left):int(window_right), 0] / 4 * 3
                
                # Output predicted large balcony
                img[int(Lbal_top):int(window_bot), 0:orig_width, 0] = numpy.ones((int(window_bot) - int(Lbal_top), orig_width)) * 255
                img[int(Lbal_top):int(window_bot), 0:orig_width, 2] = img[int(Lbal_top):int(window_bot), 0:orig_width, 2]
                img[int(Lbal_top):int(window_bot), 0:orig_width, 1] = img[int(Lbal_top):int(window_bot), 0:orig_width, 1]
                
                union = 0
                for c in range(int(window_top_truth), int(window_bot_truth)):
                    for d in range(int(window_left_truth), int(window_right_truth)):
                        if (c > window_top and c <= window_bot and d > window_left and d <= window_right):
                            union += 1
                
                tot_union += union
                facade_error += ((window_bot - window_top) * (window_right - window_left) + (window_bot_truth - window_top_truth) * (window_right_truth - window_left_truth) - 2 * union)
                #print(facade_error)
        
        #print((facade_error / 2 + tot_union) / (orig_height * orig_width))
        total_err += (facade_error / (orig_height * orig_width))

        img_tmp = Image.fromarray(img.astype(numpy.uint8))
        img_tmp.save("out/{}".format(os.path.split(image)[1]))
        
        img_num += 1
      
    total_err /= img_num
    
    print(total_err)

test()
