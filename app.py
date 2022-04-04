import os
import glob
import shutil
# Importing the important libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tensorflow import keras
import uuid
import urllib

from flask import Flask, request, abort, jsonify, send_from_directory

cnn_model = keras.models.load_model('cnn_SEI_IT_back_classifier_02_paper_wo_tripleloss.h5')

pathology_dictionary = {0:"Hemiplegic", 1:"Diplegic", 2:"Neuropathic", 3:"Normal", 4:"Parkinson"}

api = Flask(__name__,  static_url_path='', static_folder='static')

def download_video_url(req_data):
    video_name = str(uuid.uuid1()) + ".mp4"
    ret = req_data["video_url"]
    ret = "youtube-dl -o './video_uploads/"+video_name+" "+ ret
    os.system(ret)
    return video_name

def get_video_name(video_url):
    video_name = video_url.split('/')[-1]
    video_name = video_name.replace("%2F", "_")
    video_name = video_name.replace("%3A", "_")
    video_name = video_name.split("?")[0]
    return video_name



@api.route("/")
def hello_world():
	return "Hello World! I am an Openpose Server!"


@api.route("/analyze_video", methods=["POST"])
def post_file():

    # download the video from the url
    req_data = request.get_json()
    video_name = download_video_url(req_data)
    video_path = "./video_uploads/"+video_name

    # run openpose on the video and save the json file

    # Load the model

    # preprocess the json that was saved

    # put preprocees input into model

    # get prediction

    # create dictionary to return the prediction as json

    ret_pred = {
        "steppage gait": "",
        "STL": ""
    }

    return jsonify(ret_pred)

@api.route("/generate_sei", methods=["POST"])
def sei_generation():

    # download the video from the url
    req_data = request.get_json()
    
    video_name = download_video_url(req_data)
    video_path = "./video_uploads/"+video_name
    # run openpose on the video and save skeletons on skeletons/videoname folder

    # compute the sei and save it in static/sei folder

    sei_file_name = "" #filename for the saved SEIs goes here
    # return the URL for the 
    sei_url = "/static/sei/"+sei_file_name
    ret = {
        "sei_url": sei_url
    }

    return jsonify(ret)

def download_image_url(req_data):
    image_url = req_data['image_url']
    image_name = str(uuid.uuid1()) + ".jpg"
    image_path =  "./image_uploads/"+image_name
    urllib.request.urlretrieve(image_url, image_path)
    return image_name

@api.route("/predict_SEI", methods=["POST"])
def sei_prediction():

    # get url and image
    req_data = request.get_json()
    image_name = download_image_url(req_data)

    image_path = "./image_uploads/"+image_name

    # open image and preprocess
    input_image = np.array(Image.open(image_path))
    input_image = input_image.reshape((1, 224,224, 1))
    input_image = input_image/255.0
    cnn_output = cnn_model.predict(input_image)
    prediction = np.argmax(cnn_output)

    predicted_pathology = pathology_dictionary[prediction]

    ret_dic = {
        "prediction": predicted_pathology
    }

    for i in range(5):
        ret_dic[pathology_dictionary[i]] = str(cnn_output[0][i])


    return jsonify(ret_dic)

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', port=8000)