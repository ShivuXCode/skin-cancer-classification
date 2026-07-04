import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Libraries loaded")
BASE_PATH = os.getcwd()
IMG_DIR = os.path.join(BASE_PATH, "images")
CSV_PATH = os.path.join(BASE_PATH, "GroundTruth.csv")

print("Images:", IMG_DIR)
print("CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

print(df.head())
print("Columns:", df.columns.tolist())

# Your class columns in CAPS
class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Convert one-hot encoded labels into single class label
df["label"] = df[class_cols].idxmax(axis=1)

print("7-Class Distribution:")
print(df["label"].value_counts())

df["image_path"] = df["image"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))
# Keep only valid files
df = df[df["image_path"].apply(os.path.exists)]
print("Valid images:", len(df))

MALIGNANT = ["MEL", "BCC", "AKIEC"]
BENIGN = ["NV", "BKL", "DF", "VASC"]

def to_binary(label):
    return "malignant" if label in MALIGNANT else "benign"

df["binary_label"] = df["label"].apply(to_binary)

print("Binary Distribution:")
print(df["binary_label"].value_counts())

# Binary split
train_df_bin, val_df_bin = train_test_split(
    df, test_size=0.2, stratify=df["binary_label"], random_state=42
)

# Multi-class split
train_df_mc, val_df_mc = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

print("Binary  -> Train:", train_df_bin.shape, " Val:", val_df_bin.shape)
print("7-Class -> Train:", train_df_mc.shape, " Val:", val_df_mc.shape)

IMG_SIZE = 224
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

# --- Binary ---
train_ds_bin = train_gen.flow_from_dataframe(
    train_df_bin,
    x_col="image_path",
    y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds_bin = val_gen.flow_from_dataframe(
    val_df_bin,
    x_col="image_path",
    y_col="binary_label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- Multi-class ---
train_ds_mc = train_gen.flow_from_dataframe(
    train_df_mc,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds_mc = val_gen.flow_from_dataframe(
    val_df_mc,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

def build_base():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    return base, x
base_bin, x_bin = build_base()
out_bin = Dense(1, activation="sigmoid")(x_bin)
model_bin = Model(inputs=base_bin.input, outputs=out_bin)

model_bin.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model_bin.summary()

base_mc, x_mc = build_base()
out_mc = Dense(7, activation="softmax")(x_mc)
model_mc = Model(inputs=base_mc.input, outputs=out_mc)

model_mc.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_mc.summary()

callbacks_bin = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1),
    ModelCheckpoint("best_ham10000_binary.h5", monitor="val_loss", save_best_only=True)
]

callbacks_mc = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1),
    ModelCheckpoint("best_ham10000_multiclass.h5", monitor="val_loss", save_best_only=True)
]

print("====== TRAINING BINARY ======")
history_bin = model_bin.fit(
    train_ds_bin,
    validation_data=val_ds_bin,
    epochs=20,
    callbacks=callbacks_bin
)

loss_b, acc_b, auc_b = model_bin.evaluate(val_ds_bin)
print(f"[BINARY] Val Accuracy: {acc_b*100:.2f}%")
print(f"[BINARY] Val AUC: {auc_b:.4f}")
# Save Binary Model
binary_model_path = "ham10000_binary_model.h5"
model_bin.save(binary_model_path)
print(f"✅ Binary model saved as: {binary_model_path}")

print("====== TRAINING 7-CLASS ======")
history_mc = model_mc.fit(
    train_ds_mc,
    validation_data=val_ds_mc,
    epochs=20,
    callbacks=callbacks_mc
)

loss_m, acc_m = model_mc.evaluate(val_ds_mc)
print(f"[7-CLASS] Val Accuracy: {acc_m*100:.2f}%")
# Save Multi-class Model
multiclass_model_path = "ham10000_multiclass_model.h5"
model_mc.save(multiclass_model_path)
print(f"✅ Multi-class model saved as: {multiclass_model_path}")

plt.figure(figsize=(14,5))

# Binary
plt.subplot(1,2,1)
plt.plot(history_bin.history["accuracy"], label="Train Acc")
plt.plot(history_bin.history["val_accuracy"], label="Val Acc")
plt.title("Binary Accuracy")
plt.legend()

# Multi-class
plt.subplot(1,2,2)
plt.plot(history_mc.history["accuracy"], label="Train Acc")
plt.plot(history_mc.history["val_accuracy"], label="Val Acc")
plt.title("7-Class Accuracy")
plt.legend()

plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to preprocess single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

print("Prediction helper ready")

print("\nChoose prediction type:")
print("1 -> Binary (Benign / Malignant)")
print("2 -> Multi-class (7 Classes)")

choice = input("Enter your choice (1 or 2): ").strip()

img_path = input("\nEnter full image path (e.g., C:/Users/manis/Vikram/images/ISIC_0024306.jpg): ").strip()

if not os.path.exists(img_path):
    print("❌ Image file not found. Check the path.")
else:
    print("✅ Image found. Running prediction...")

# Preprocess image
img_input = preprocess_image(img_path)

# Show the image
img_show = image.load_img(img_path)
plt.imshow(img_show)
plt.axis("off")
plt.show()

# --------- BINARY PREDICTION ---------
if choice == "1":
    pred = model_bin.predict(img_input)[0][0]
    
    if pred >= 0.5:
        print(f"\n🧪 Binary Prediction: MALIGNANT ({pred*100:.2f}%)")
    else:
        print(f"\n🧪 Binary Prediction: BENIGN ({(1-pred)*100:.2f}%)")

# --------- MULTI-CLASS PREDICTION ---------
elif choice == "2":
    class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    
    preds = model_mc.predict(img_input)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    print("\n🧪 Multi-class Prediction:")
    for cls, p in zip(class_names, preds):
        print(f"{cls}: {p*100:.2f}%")

    print(f"\n✅ Final Prediction: {predicted_class} ({confidence*100:.2f}%)")

# --------- INVALID INPUT ---------
else:
    print("❌ Invalid choice. Enter 1 for Binary or 2 for Multi-class.")





