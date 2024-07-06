from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = load_model('cnn_deepfake_image.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(256, 256))
    img_tensor = image.img_to_array(img)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    predicted = model.predict(img_tensor)
    predicted_class = tf.argmax(predicted, axis=1).numpy()[0]
    return predicted_class

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict the uploaded image
        predicted_class = predict_image(filepath)
        
        # Determine prediction label and class for CSS
        prediction_label = 'Fake' if predicted_class == 1 else 'Real'
        prediction_class = 'fake' if predicted_class == 1 else 'real'
        
        # Return prediction along with filename and prediction class
        return jsonify({'filename': filename, 'prediction': prediction_label, 'prediction_class': prediction_class}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
