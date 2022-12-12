from flask import Flask, render_template, request, flash, redirect
import urllib.request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
import numpy as np

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "akdschssjkdv"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_DIMENSIONS'] = 16 * 1024 * 1024

ALLOWED_EXT = set(['png', 'jpg', 'jpeg'])
def allowedFiles(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# # print(tf.__version__)
# BATCH_SIZE = 10
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# cnnModel = load_model("models/DenseNet.h5")

# ds_train_ = tf.keras.utils.image_dataset_from_directory(
#     "train",
#     labels='inferred',
#     #label_mode='binary',
#     image_size=[200, 200],
#     interpolation='nearest',
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# # Data Pipeline
# def convert_to_float(image, label):
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     return image, label

# ds_train = (
#     ds_train_
#     .map(convert_to_float)
#     .cache()
#     .prefetch(buffer_size=AUTOTUNE)
# )

# img = tf.keras.utils.load_img("parrot.jpg", target_size=(200, 200))
# img_array = tf.keras.utils.img_to_array(img=img)
# img_array = tf.expand_dims(img_array, 0)

# class_names = ds_train_.class_names

# predictions = cnnModel.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )



@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("index.html")




@app.route("/upload", methods=['POST'])
def imgUpload():
    if 'file' not in request.files:
        flash("No file! 😕")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No image selected for uploading! 😕")
        return redirect(request.url)
    if file and allowedFiles(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash("Image uploaded successfully! 🚀")
        return render_template("Densenet.html", filename=filename)
    else:
        flash("Allowed file types are: png, jpg, jpeg 😌")
        return redirect(request.url)


@app.route("/chooseModel", methods=["POST", "GET"])
def chooseModel():
    return render_template("choose.html")

@app.route("/cnnModel", methods=["POST", "GET"])
def cnnModel():
   pass

@app.route("/denseNetModel", methods=["POST", "GET"])
def denseNetModel():
    if 'file' not in request.files:
        flash("No file! 😕")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No image selected for uploading! 😕")
        return redirect(request.url)
    if file and allowedFiles(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash("Image uploaded successfully! 🚀")
        return render_template("Densenet.html", filename=filename)
    else:
        flash("Allowed file types are: png, jpg, jpeg 😌")
        return render_template("Densenet.html")

if __name__ == "__main__":
    app.run(debug=True)