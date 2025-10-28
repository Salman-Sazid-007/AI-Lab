
# ------------------------------------------
# 📦 Import Necessary Libraries
# ------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------
# 🧩 Load and Preprocess CIFAR-10 Dataset
# ------------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)

# Normalize pixel values (0–1)
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# CIFAR-10 has 10 classes (0–9)
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# ------------------------------------------
# 🔁 Data Augmentation
# ------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)

# ------------------------------------------
# 🧠 Build CNN Model
# ------------------------------------------
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Fully Connected
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

# ------------------------------------------
# ⚙️ Compile the Model
# ------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------
# 🚀 Train the Model
# ------------------------------------------
epochs = 30
batch_size = 64

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    steps_per_epoch=X_train.shape[0] // batch_size,
    verbose=1
)

# ------------------------------------------
# 🧪 Evaluate Model
# ------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# ------------------------------------------
# 📊 Plot Training Curves
# ------------------------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.savefig("cifar10_augmentation_accuracy.png")
plt.show()

# ------------------------------------------
# 🖼️ Visualize Predictions
# ------------------------------------------
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

plt.figure(figsize=(12,12))
for i in range(16):
    idx = np.random.randint(0, len(X_test))
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Pred: {class_names[y_pred[idx]]}\nTrue: {class_names[int(y_test[idx])]}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("cifar10_predictions.png")
plt.show()

# ------------------------------------------
# 🧾 Classification Report
# ------------------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))
