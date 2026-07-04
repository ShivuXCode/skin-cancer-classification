import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# 1. Setup Local Paths
BASE_PATH = os.getcwd()
ARCHIVE_DIR = os.path.join(BASE_PATH, "archive")
IMG_DIR = os.path.join(ARCHIVE_DIR, "images")
CSV_PATH = os.path.join(ARCHIVE_DIR, "GroundTruth.csv")

print("Checking paths...")
print(f"Images directory exists: {os.path.exists(IMG_DIR)}")
print(f"CSV file exists: {os.path.exists(CSV_PATH)}")

# 2. Data Loading & Preprocessing
df = pd.read_csv(CSV_PATH)

# The classes are one-hot encoded in this CSV version
class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
df["label"] = df[class_cols].idxmax(axis=1)

# Append full path and .jpg extension to the image column
# Notice we assume the images are directly inside 'archive/images'
df["image_path"] = df["image"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))

# Filter out rows if the image file doesn't exist locally
df = df[df["image_path"].apply(os.path.exists)]
print(f"\nTotal valid images found: {len(df)}")

if len(df) == 0:
    raise ValueError("No valid images found! Please make sure your images are directly inside the 'archive/images' directory.")

# Create Binary Labels (Malignant vs Benign)
MALIGNANT = ["MEL", "BCC", "AKIEC"]
df["binary_label"] = df["label"].apply(lambda x: "malignant" if x in MALIGNANT else "benign")

print("\nBinary Distribution:")
print(df["binary_label"].value_counts())

# Train/Val Split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["binary_label"], random_state=42
)

# 3. Image Generators & Class Weights
IMG_SIZE = 224
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40, 
    zoom_range=0.2,
    width_shift_range=0.2, 
    height_shift_range=0.2,
    horizontal_flip=True, 
    vertical_flip=True, 
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_dataframe(
    train_df, x_col="image_path", y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE), class_mode="binary",
    batch_size=BATCH_SIZE, shuffle=True
)

val_ds = val_gen.flow_from_dataframe(
    val_df, x_col="image_path", y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE), class_mode="binary",
    batch_size=BATCH_SIZE, shuffle=False
)

# Calculate Class Weights to penalize the model heavily if it misses cancer (Malignant)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['binary_label']),
    y=train_df['binary_label']
)
class_indices = train_ds.class_indices 
class_weight_dict = {
    class_indices['benign']: class_weights_array[np.where(np.unique(train_df['binary_label']) == 'benign')[0][0]],
    class_indices['malignant']: class_weights_array[np.where(np.unique(train_df['binary_label']) == 'malignant')[0][0]]
}
print("\nClass Weights:", class_weight_dict)

# 4. Model Building Function
def build_model(base_model_func, model_name):
    print(f"\n---> Building {model_name}...")
    base = base_model_func(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = True # Unfreeze for fine-tuning
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base.input, outputs=out)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# 5. Training Loop for 3 Models
models_to_test = {
    "EfficientNetB0": EfficientNetB0,
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2
}

histories = {}
results = {}
EPOCHS = 10 # Set higher (e.g., 20 or 30) for better results

for name, model_func in models_to_test.items():
    print(f"\n======================================")
    print(f"        TRAINING {name}         ")
    print(f"======================================")
    
    model = build_model(model_func, name)
    
    early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f"best_{name}_binary.h5", monitor="val_auc", mode="max", save_best_only=True)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    histories[name] = history.history
    loss, acc, auc = model.evaluate(val_ds, verbose=0)
    results[name] = {"Accuracy": acc, "AUC": auc}
    print(f"\n[Finished {name}] -> Val Acc: {acc:.4f} | Val AUC: {auc:.4f}\n")

# 6. Plotting the Comparisons
plt.figure(figsize=(18, 5))

# Plot 1: Validation AUC
plt.subplot(1, 3, 1)
names = list(results.keys())
aucs = [res["AUC"] for res in results.values()]
plt.bar(names, aucs, color=['blue', 'orange', 'green'])
plt.title('Validation AUC (Higher is better)')
plt.ylim(0.5, 1.0)
for i, v in enumerate(aucs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

# Plot 2: Validation Accuracy
plt.subplot(1, 3, 2)
accs = [res["Accuracy"] for res in results.values()]
plt.bar(names, accs, color=['blue', 'orange', 'green'])
plt.title('Validation Accuracy')
plt.ylim(0.5, 1.0)
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

# Plot 3: Learning Curves (AUC over epochs)
plt.subplot(1, 3, 3)
for name, hist in histories.items():
    plt.plot(hist['val_auc'], marker='o', label=f"{name} Val AUC")
plt.title('Validation AUC over Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig("model_comparison_results.png")
print("\n✅ Training complete! Results plotted and saved as 'model_comparison_results.png'.")
plt.show()
