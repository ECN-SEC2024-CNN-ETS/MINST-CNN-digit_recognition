from utility import DrawNumber as imgUtils
from utility import ImgProcessing as imgProc
from keras.models import load_model
from numpy import argmax, invert


def run_program():
    imgUtils.draw_number()   # Asks the user to input an image and saving it

    img = imgProc.load_image('utility/resources/handwritten_numbers_resize.png')  # Loading the hand drawn image

    model = load_model('model/MODEL_SEC2024.h5')  # Loading the generated model
    predict_value = model.predict(img)  # Using the model to predict the number drawn by the user

    digit = argmax(predict_value)   # Getting the maximum value
    print("Réponse du modele : ")
    print(digit)


#   Main program
run_program()
