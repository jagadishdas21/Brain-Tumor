from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


logging.basicConfig(level=logging.DEBUG)


model_path = os.path.join(os.getcwd(), 'model.keras')
model = load_model(model_path)

def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img.reshape(1, 128, 128, 3) / 255.0
    return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Prediction request received")

  
        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        logging.info(f"Received file: {file.filename}")


        if file.filename == '':
            logging.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
  
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        logging.info("Image loaded successfully")
        
        preprocessed_image = preprocess_image(image)
        logging.debug(f"Preprocessed image shape: {preprocessed_image.shape}")


        prediction = model.predict(preprocessed_image)
        logging.info(f"Raw prediction: {prediction}")


        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        logging.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        return jsonify({'prediction': int(predicted_class), 'confidence': float(confidence)}), 200
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500  # Handle unexpected errors

if __name__ == '__main__':
    # Ensure that the app runs on Render or any platform with specified port
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
