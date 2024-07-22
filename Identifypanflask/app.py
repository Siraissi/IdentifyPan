import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FEEDBACK_FOLDER'] = 'feedback'

model_path = r'C:\Users\Siraissi\Documents\GitHub\IdentifyPan\trainamento\modelo_classificacao_pan.keras'
model = load_model(model_path)

def classify_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = classify_image(filepath)
            classification = "Pan" if prediction >= 0.5 else "Não Pan"
            return render_template('result.html', filename=filename, classification=classification, prediction=prediction)
    return render_template('upload.html')

@app.route('/feedback/<filename>/<classification>/<prediction>', methods=['POST'])
def feedback(filename, classification, prediction):
    feedback = request.form['feedback']
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if feedback == 'correto':
        if classification == "Pan":
            shutil.move(original_path, os.path.join(app.config['FEEDBACK_FOLDER'], 'pan', filename))
        else:
            shutil.move(original_path, os.path.join(app.config['FEEDBACK_FOLDER'], 'not_pan', filename))
    else:
        if classification == "Pan":
            shutil.move(original_path, os.path.join(app.config['FEEDBACK_FOLDER'], 'not_pan', filename))
        else:
            shutil.move(original_path, os.path.join(app.config['FEEDBACK_FOLDER'], 'pan', filename))
    return redirect(url_for('upload_image'))

if __name__ == "__main__":
    app.run(debug=True)
