from flask import Flask, render_template, request, flash, redirect
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import urllib.request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
import numpy as np

UPLOAD_FOLDER = 'static/uploads/'


app = Flask(__name__)
app.config['SECRET_KEY'] = 'sjdovbk'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_DIMENSIONS'] = 16 * 1024 * 1024

ALLOWED_EXT = set(['png', 'jpg', 'jpeg'])
def allowedFiles(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

# print(tf.__version__)
BATCH_SIZE = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE


ds_train_ = tf.keras.utils.image_dataset_from_directory(
    "train",
    labels='inferred',
    #label_mode='binary',
    image_size=[200, 200],
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

def findBirdDenseNet(filename):
    print("i am here")
    print(filename)
    img = tf.keras.utils.load_img("static/uploads/"+filename, target_size=(200, 200))
    img_array = tf.keras.utils.img_to_array(img=img)
    img_array = tf.expand_dims(img_array, 0)
    print(img_array)
    class_names = ds_train_.class_names
    denseNetModel = load_model("models/DenseNet.h5")
    predictions = denseNetModel.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return(
        "This bird is most likely a {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def findBirdCNN(filename):
    print("i am here")
    print(filename)
    img = tf.keras.utils.load_img("static/uploads/"+filename, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img=img)
    img_array = tf.expand_dims(img_array, 0)
    print(img_array)
    class_names = ds_train_.class_names
    denseNetModel = load_model("models/birdsCNN.h5")
    predictions = denseNetModel.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return(
        "This bird is most likely a {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


# findBird("hawk.jpeg")


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
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], file.filename))
        print("eyeyeyeyey")
        text = findBirdDenseNet(file.filename)
        return text
    return render_template("Densenet.html", form=form)


@app.route("/CNNModel", methods=["POST", "GET"])
def CNNModel():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], file.filename))
        print("eyeyeyeyey")
        text = findBirdCNN(file.filename)
        return text
    return render_template("CNN.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)