import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Original Train Shape:", X_train.shape)
print("Original Test Shape :", X_test.shape)

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print("CNN Train Shape:", X_train.shape)
print("CNN Test Shape :", X_test.shape)


inputs = Input((28,28,1))

x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs, name="CNN_MNIST")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)


y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
print("✅ Test Accuracy:", round(float(acc)*100, 2), "%")


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

plt.savefig("cnn_accuracy.png")
plt.show()


plt.figure(figsize=(15,15))
rows, cols = 5, 5

for i in range(25):
    plt.subplot(rows, cols, i+1)
    pred = y_pred[i]
    plt.title(f"True: {y_test[i]}, Pred: {pred}", fontsize=12)
    plt.imshow(X_test[i].reshape(28,28), cmap="gray")
    plt.axis('off')

plt.savefig("cnn_results.png")
plt.tight_layout()
plt.show()
