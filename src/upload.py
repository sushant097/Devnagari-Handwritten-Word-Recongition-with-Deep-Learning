# -*- coding: utf-8 -*-

import os

from flask import Flask, request, render_template, send_from_directory
from datetime import datetime
from main import infer_by_web

__author__ = 'Sushant'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # project abs path

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_page", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    # folder_name = request.form['uploads']
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    option = request.form.get('optionsPrediction')
    print("Selected Option:: {}".format(option))
    for upload in request.files.getlist("file"):
        print(upload)
        filename = upload.filename
        print("{} is the file name".format(upload.filename))
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')#upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        result, probability = predict_image(destination, option)
        #print("Prediction: ", result)
    # return send_from_directory("images", filename, as_attachment=True)
    print("Send File Name to html: ", filename)
    return render_template("complete.html", image_name=filename, result=[result,probability]) # , probability


def predict_image(path, type):
    print(path)
    return infer_by_web(path, type)



'''
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
 

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)
'''


if __name__ == "__main__":
    app.run(port=5555, debug=True)
