import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'archive')
IMG_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'GroundTruth.csv')

# Load Models
print("Loading models...")
binary_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_EfficientNetB0_binary.h5'))
malignant_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'hierarchical_malignant_model.h5'))
benign_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'hierarchical_benign_model.h5'))

# Classes mapping
MALIGNANT_CLASSES = ["AKIEC", "BCC", "MEL"]
BENIGN_CLASSES = ["BKL", "DF", "NV", "VASC"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load Ground Truth
df = pd.read_csv(CSV_PATH)
class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
df["true_label"] = df[class_cols].idxmax(axis=1)

MALIGNANT_SET = {"MEL", "BCC", "AKIEC"}
df["true_binary"] = df["true_label"].apply(lambda x: "Malignant" if x in MALIGNANT_SET else "Benign")

# Take a random sample of 200 images to test
test_df = df.sample(n=200, random_state=42)

binary_correct = 0
specific_correct = 0
total = 0

print("Running predictions on 200 random images...\n")

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_name = row['image']
    img_path = os.path.join(IMG_DIR, f"{img_name}.jpg")
    
    if not os.path.exists(img_path):
        continue
        
    try:
        img_input = preprocess_image(img_path)
    except Exception:
        continue
        
    # Binary Prediction
    bin_pred = binary_model.predict(img_input, verbose=0)[0][0]
    is_malignant = bin_pred >= 0.5
    pred_binary = "Malignant" if is_malignant else "Benign"
    
    if pred_binary == row['true_binary']:
        binary_correct += 1
        
    # Sub-class Prediction
    if is_malignant:
        sub_preds = malignant_model.predict(img_input, verbose=0)[0]
        pred_specific = MALIGNANT_CLASSES[np.argmax(sub_preds)]
    else:
        sub_preds = benign_model.predict(img_input, verbose=0)[0]
        pred_specific = BENIGN_CLASSES[np.argmax(sub_preds)]
        
    if pred_specific == row['true_label']:
        specific_correct += 1
        
    total += 1

print("\n--- RESULTS ---")
print(f"Total Images Evaluated: {total}")
print(f"Binary Accuracy (Benign vs Malignant): {binary_correct / total * 100:.2f}%")
print(f"Specific Type Accuracy (Exact 7 classes): {specific_correct / total * 100:.2f}%")
