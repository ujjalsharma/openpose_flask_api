import os
import glob
from sys import platform
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
import cv2

from flask import Flask, request, abort, jsonify, send_from_directory

cnn_model = keras.models.load_model('cnn_SEI_IT_back_classifier_02_paper_wo_tripleloss.h5')

pathology_dictionary = {0:"Hemiplegic", 1:"Diplegic", 2:"Neuropathic", 3:"Normal", 4:"Parkinson"}

api = Flask(__name__,  static_url_path='', static_folder='static')

def download_video_url(req_data):
    video_name = str(uuid.uuid1()) + ".mp4"
    video_url = req_data["video_url"]
    video_path = "./video_uploads/"+video_name
    # ret = "youtube-dl -o './video_uploads/"+video_name+" "+ ret
    # os.system(ret)
    
    urllib.request.urlretrieve(video_url, video_path)
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
def analyze_video_json():

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


def download_image_url(req_data):
    image_url = req_data['image_url']
    image_name = str(uuid.uuid1()) + ".jpg"
    image_path =  "./image_uploads/"+image_name
    urllib.request.urlretrieve(image_url, image_path)
    return image_name

def convert_to_black_and_white(img):
    I = img.shape[0]
    J = img.shape[1]
    new_img = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            if (img[i][j]!=0).any():
                new_img[i][j]=255
    return new_img

def convert_to_sei(skeletons_folder_path, sei_folder_path):
    list = os.listdir(skeletons_folder_path) # dir is your directory path
    number_files = len(list)
    
    img = cv2.imread(skeletons_folder_path+"/"+list[0])
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sei_img = np.zeros(g_img.shape)
    
    for image_file_name in list:
        n_img = cv2.imread(skeletons_folder_path+"/"+image_file_name)
        #n_g_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
        n_g_img = convert_to_black_and_white(n_img)
        sei_img  = sei_img+n_g_img
        
    sei_img = sei_img/number_files;
    cv2.imwrite(sei_folder_path, sei_img)

def predict_sei(image_path):
    # open image and preprocess
    im = Image.open(image_path)
    im = im.resize((224,224))
    input_image = np.array(im)
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

@api.route("/analyze_video_CNN", methods=["POST"])
def cnn_video_prediction():
    # download the video from the url
    req_data = request.get_json()
    
    video_name = download_video_url(req_data)
    video_path = "../video_uploads/"+video_name
    skeleton_folder_path = "../static/saved_skeletons/"+video_name[:-4]+"/"
    os.mkdir("static/saved_skeletons/"+video_name[:-4])
    # run openpose on the video path
    commnd = "!(cd openpose/ && bin/OpenPoseDemo.exe --video "+video_path+" --net_resolution 160x160 --write_images "+skeleton_folder_path+" --disable_blending)"
    
    if platform == "linux" or platform == "linux2":
        commnd = "!(cd openpose/ && ./build/examples/openpose/openpose.bin --video "+video_path+" --net_resolution 160x160 --write_images "+skeleton_folder_path+" --disable_blending)"
    os.system(commnd)

    # filter only gait cycle and process the images

    # saved the seis
    sei_path = "./static/seis/"+video_name[:-4]+".jpg"
    skeleton_folder_path = "static/saved_skeletons/"+video_name[:-4]+"/"
    convert_to_sei(skeleton_folder_path, sei_path)

    return predict_sei(sei_path)


@api.route("/predict_SEI", methods=["POST"])
def sei_prediction():

    # get url and image
    req_data = request.get_json()
    image_name = download_image_url(req_data)

    image_path = "./image_uploads/"+image_name

    return predict_sei(image_path)    

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', port=8000)