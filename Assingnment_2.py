import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# generate dataset
def generate_data(n=500, xmin=-10, xmax=10):
    x = np.linspace(xmin, xmax, n)
    y = 5 * x**2 + 10 * x - 2
    return x.reshape(-1, 1), y.reshape(-1, 1)

x, y = generate_data()

# the model

inputs = Input(shape=(1,), name="Input_Layer")
h1 = Dense(64, activation="relu", name="Hidden_Layer_1")(inputs)
h2 = Dense(128, activation="relu", name="Hidden_Layer_2")(h1)
outputs = Dense(1, name="Output_Layer")(h2)

model = Model(inputs, outputs, name="FCFNN")
model.compile(optimizer="adam", loss="mse")

# train model

history = model.fit(x, y, epochs=200, batch_size=32, verbose=1)

y_pred = model.predict(x)

plt.figure(figsize=(8,6))
plt.plot(x, y, label="Original f(x)", color="blue")
plt.plot(x, y_pred, label="Predicted f(x)", color="red", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("FCFNN Approximation of f(x) = 5x^2 + 10x - 2")
plt.legend()
plt.grid(True)
plt.savefig("results.png")  
plt.show()

