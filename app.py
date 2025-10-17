from flask import Flask, render_template, request
import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import joblib


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# loading objects and models
logistic_reg = joblib.load("logistic_regration.joblib")
svc = joblib.load("SVC_model.joblib")
pca = joblib.load("PCA_N_reduce.joblib")

#importind class indecies
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_label = {v: k for k, v in class_indices.items()}


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False
vgg16 = Model(inputs=base_model.input, outputs=base_model.output)


def preprocess_img(file_path):
    """Load and preprocess an image like in training."""
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Same as training
    return img_array


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods = ['post'])
def predict():

    if 'file' not in request.files:
        return render_template('home.html', prediction="No file uploaded.")
    
    file = request.files['file']
    model_choice = request.form.get('model')
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img_array = preprocess_img(file_path)
    features = vgg16.predict(img_array)
    flat_feature = features.reshape(1, -1)
    pca_transformed = pca.transform(flat_feature)


    if model_choice == "svc":
        y_pred = svc.predict(pca_transformed)
        print("Predicted class:", y_pred)
    elif model_choice == "logistic":
        y_pred = logistic_reg.predict(pca_transformed)
        print("Predicted class:", y_pred)
    else :
        return render_template('home.html', prediction="invalid model selection!")
    
    prediction_class = int(y_pred[0])
    prediction = idx_to_label.get(prediction_class, "Unknown")


    return render_template('home.html', prediction=f"The image is classified as: {prediction}")



if __name__ == "__main__":
    app.run(debug=True)