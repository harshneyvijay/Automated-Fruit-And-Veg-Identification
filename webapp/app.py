from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model("trained_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Model prediction function
def model_prediction(image_path):
    image = Image.open(image_path).resize((64, 64))
    input_arr = np.expand_dims(np.array(image), axis=0)
    prediction = model.predict(input_arr)
    return labels[np.argmax(prediction)]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
        
        image_path = os.path.join("static", file.filename)
        file.save(image_path)
        result = model_prediction(image_path)

        return render_template("predict.html", prediction=result, uploaded_image=image_path)

    return render_template("predict.html", prediction=None, uploaded_image=None)

if __name__ == "__main__":
    app.run(debug=True)
