
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import flask
from flask import Flask, request, jsonify, render_template
import imageio as Image
import base64
import io
import os
import cv2



file_path = 'C:/Users/Shahid Sanghar/Desktop/Datasets/dogs-vs-cats'
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

model = tf.keras.models.load_model('static/models')

# pickle.dump(model, open('best_model.hdf5', 'wb'))


@app.route('/')
def home():
    return render_template('index.html', prediction_text="")


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on html gui
    '''
    upload_file = request.files['Select_picture']

    filename = upload_file.filename
    print(filename)

    ext = filename.split('.')[-1]
    print('The extension of the filename =', ext)
    if ext.lower() in ['png', 'jpg', 'jpeg']:
        # saving the image
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        print('File saved sucessfully')

    # Use the uploaded image

        # load the test image
        image = cv2.imread(path_save)
        output = image.copy()
        image = cv2.resize(image, (128, 128))

        # scale the pixels
        image = image.astype('float') / 255.0

        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))

        # predict
        pred = model.predict(image)
        animal = 'Cat' if pred[0][0] > pred[0][1] else 'Dog'
        print(animal)

        return render_template('index.html', prediction_text=animal)


if __name__ == '__main__':
    app.run(debug=True)



