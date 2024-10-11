from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load your saved model
model = load_model('my_model.keras')

def preprocess_image(image):
    img = image.resize((128, 128))  # Adjust size as per your model
    img = np.array(img)
    img = img.reshape(1, 128, 128, 3) / 255.0  # Normalize and reshape
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Open the image
        image = Image.open(io.BytesIO(file.read()))
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({'prediction': int(predicted_class)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
