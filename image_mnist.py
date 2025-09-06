# ===============================
# Import Section
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score

# ===============================
# Load Dataset Section
# ===============================
data = np.load("dataset_mnist_own.npz") 
X_train = data['trainX']
y_train = data['trainY']
X_test  = data['testX']
y_test  = data['testY']


print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ===============================
# Normalize + Flatten Section
# ===============================
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

print("Flattened Train Shape:", X_train.shape)
print("Flattened Test Shape :", X_test.shape)

# ===============================
# Neural Network Section (Improved FCNN)
# ===============================
inputs = Input((784,))
x = Dense(256, activation="relu")(inputs)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)

# ===============================
# Compile Section
# ===============================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Model Fitting Section
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=100,  
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ===============================
# Evaluate Section
# ===============================
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
print("✅ Test Accuracy:", round(float(acc)*100, 2), "%")

# ===============================
# Plot Loss & Accuracy
# ===============================
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curves')
plt.legend()

plt.show()

# ===============================
# Predict and Plot Few Images
# ===============================
plt.figure(figsize=(15,15))
n_show = min(25, len(X_test))
rows, cols = 5, 5

for i in range(n_show):
    plt.subplot(rows, cols, i+1)
    sample = X_test[i].reshape(1, 784)  
    pred = model.predict(sample, verbose=0).argmax(axis=1)[0]
    plt.title(f"Pred: {pred}", fontsize=14)
    plt.imshow(X_test[i].reshape(28,28), cmap="gray")
    plt.axis('off')

plt.tight_layout()
plt.show()
