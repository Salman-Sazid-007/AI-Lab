# -----------------------------
# Import Libraries
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load and Preprocess Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape for CNN input
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert multi-class labels (0–9) into binary:
# even -> 0, odd -> 1
y_train_binary = np.where(y_train % 2 == 0, 0, 1)
y_test_binary = np.where(y_test % 2 == 0, 0, 1)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Example labels: {y_train_binary[:10]}")

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.summary()

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train, y_train_binary,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test_binary),
    verbose=1
)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test_binary, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")

print(classification_report(y_test_binary, y_pred, target_names=['Even', 'Odd']))

# -----------------------------
# Plot Accuracy and Loss
# -----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig("mnist_even_odd_accuracy.png")
plt.show()

# -----------------------------
# Show Predictions
# -----------------------------
plt.figure(figsize=(10,10))
for i in range(16):
    idx = np.random.randint(0, len(X_test))
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    pred_label = "Odd" if y_pred[idx] == 1 else "Even"
    true_label = "Odd" if y_test_binary[idx] == 1 else "Even"
    plt.title(f"P:{pred_label} | T:{true_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("mnist_even_odd_results.png")
plt.show()
