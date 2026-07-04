import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Models
print("Loading binary model...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')

# Paths
binary_model_path = os.path.join(MODELS_DIR, 'best_EfficientNetB0_binary.h5')

# Load into memory
try:
    binary_model = tf.keras.models.load_model(binary_model_path)
    print("Binary model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(img_path):
    # EfficientNetB0 expects 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Preprocess
            img_input = preprocess_image(filepath)
            
            # 2. Binary Prediction
            bin_pred = binary_model.predict(img_input, verbose=0)[0][0]
            is_malignant = bin_pred >= 0.5
            
            primary_label = "Malignant" if is_malignant else "Benign"
            primary_confidence = float(bin_pred) if is_malignant else float(1 - bin_pred)

            # Cleanup file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'primary_diagnosis': primary_label,
                'primary_confidence': f"{primary_confidence * 100:.2f}%"
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
