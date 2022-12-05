from flask import Flask, render_template
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# print(tf.__version__)
BATCH_SIZE = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE
cnnModel = load_model("models/DenseNet.h5")

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

img = tf.keras.utils.load_img("parrot.jpg", target_size=(200, 200))
img_array = tf.keras.utils.img_to_array(img=img)
img_array = tf.expand_dims(img_array, 0)

class_names = ds_train_.class_names

predictions = cnnModel.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



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