from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load your saved model
model = load_model('model.keras')  # Ensure this file is in the root of your project directory

def preprocess_image(image):
    img = image.resize((128, 128))  # Adjust size as per your model
    img = np.array(img)
    img = img.reshape(1, 128, 128, 3) / 255.0  # Normalize and reshape
    return img

# Serve the homepage
@app.route('/')
def index():
    return render_template('index.html')  # This will serve your front-end index.html

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')  # Ensure image is in RGB format
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)

        # Log the raw prediction for debugging
        print("Raw prediction:", prediction)

        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)  # Get confidence level of prediction

        return jsonify({'prediction': int(predicted_class), 'confidence': float(confidence)}), 200  # Return prediction with confidence

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle unexpected errors

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
