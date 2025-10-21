
# Import Required Libraries

import numpy as np
import os
import random
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Dataset Paths

shirt_path = 'ResizedShirt'      
tshirt_path = 'ResizedTShirt'     

# Image size
img_height, img_width = 255, 255


# Load and Label Images

def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Could not load {filename}: {e}")
    return images, labels


# Load Dataset

shirt_images, shirt_labels = load_images_from_folder(shirt_path, 0)   # 0 = Shirt
tshirt_images, tshirt_labels = load_images_from_folder(tshirt_path, 1) # 1 = TShirt

# Combine data
X = np.array(shirt_images + tshirt_images, dtype='float32') / 255.0
Y = np.array(shirt_labels + tshirt_labels)

# Shuffle and Split
X, Y = shuffle(X, Y, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("✅ Dataset Loaded")
print("X_train:", X_train.shape, " | Y_train:", Y_train.shape)
print("X_test :", X_test.shape,  " | Y_test :", Y_test.shape)

# CNN Model Architecture (VGG-like)

inputs = Input(shape=(255, 255, 3), name="input_layer")

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs, outputs, name="Shirt_TShirt_CNN")


# Compile Model

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ⏱️ Training Phase

start_time = time.time()
history = model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1
)
train_time = time.time() - start_time

# Evaluation Phase

start_test = time.time()
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
test_time = time.time() - start_test

print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"⏱️ Training Time: {train_time:.2f} seconds")
print(f"⏱️ Testing Time : {test_time:.2f} seconds")
print(f"📦 Model Parameters: {model.count_params():,}")


# Training Graphs

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("cnn_training_results.png")
plt.show()


# Predictions and Confusion Matrix

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(Y_test, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Shirt','TShirt'], yticklabels=['Shirt','TShirt'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("cnn_confusion_matrix.png")
plt.show()


# Classification Report

report = classification_report(Y_test, y_pred_classes, target_names=['Shirt','TShirt'])
print("\nClassification Report:\n", report)


# Prediction Visualization

sample_indices = random.sample(range(len(X_test)), 20)
plt.figure(figsize=(15,10))
for i, idx in enumerate(sample_indices):
    plt.subplot(4,5,i+1)
    plt.imshow(X_test[idx])
    plt.axis('off')
    pred_label = 'TShirt' if y_pred_classes[idx]==1 else 'Shirt'
    true_label = 'TShirt' if Y_test[idx]==1 else 'Shirt'
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10)
plt.tight_layout()
plt.savefig("cnn_predictions.png")
plt.show()


# Final Model Performance Summary

print("\n FINAL SUMMARY:")
print(f" Model Name       : Shirt_TShirt_CNN")
print(f" Total Parameters : {model.count_params():,}")
print(f" Training Time    : {train_time:.2f} sec")
print(f" Testing Time     : {test_time:.2f} sec")
print(f" Test Accuracy    : {acc*100:.2f}%")
print(f" Recognition Rate : Model correctly identifies {(acc*100):.2f}% of images.")
