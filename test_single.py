import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMG_PATH = os.path.join(BASE_DIR, 'data', 'archive', 'images', 'ISIC_0024306.jpg')

print("Loading benign model...")
benign_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'hierarchical_benign_model.h5'))

BENIGN_CLASSES = ["BKL", "DF", "NV", "VASC"]

print("Preprocessing image...")
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

print("Predicting...")
preds = benign_model.predict(img_array, verbose=0)[0]

print("\nRaw Probabilities:")
for cls, prob in zip(BENIGN_CLASSES, preds):
    print(f"{cls}: {prob*100:.2f}%")
    
idx = np.argmax(preds)
print(f"\nFinal Prediction: {BENIGN_CLASSES[idx]}")
