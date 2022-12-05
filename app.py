from flask import Flask, render_template
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask(__name__)

cnnModel = load_model("models/CNN_Model.h5")

img = tf.keras.utils.load_img("parrot.jpg", target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img=img)
img_array = tf.expand_dims(img_array, 0)

predictions = cnnModel.predict(img_array)
score = tf.nn.softmax(predictions)

print(predictions[0] * 100)


@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("index.html")

@app.route("/chooseModel", methods=["POST", "GET"])
def chooseModel():
    return render_template("choose.html")

@app.route("/cnnModel", methods=["POST", "GET"])
def cnnModel():
   pass

if __name__ == "__main__":
    app.run(debug=True)