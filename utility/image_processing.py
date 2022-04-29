# make a prediction for a new image.
import numpy as geek
from numpy import argmax, invert
from ressources.draw_numbers_using_mouse import draw_number
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image

def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	#img = geek.invert(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class


def run_example():
	# load the image
    draw_number()
    
    img = load_image('images/handwritten_numbers_resize.png')
	# load model
    
    model = load_model('model/final_model_SEC_2024.h5')
	# predict the class
    predict_value = model.predict(img)
	
    digit = argmax(predict_value)
    print(digit)

# entry point, run the example
run_example()