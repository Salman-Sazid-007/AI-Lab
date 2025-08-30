from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

def main():
    # Build model
    model = build_model()
    model.compile(loss='mse', optimizer='adam')

    # Prepare data
    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_val_test()

    # Train model
    model.fit(trainX, trainY,
              validation_data=(valX, valY),
              epochs=10000,
              verbose=1)

def prepare_train_val_test():
    x, y = data_process()
    total_n = len(x)
    print(x.shape, total_n)

    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)
    test_n = total_n - train_n - val_n

    trainX = x[: train_n]
    trainY = y[: train_n]
    valX = x[train_n: train_n + val_n]
    valY = y[train_n: train_n + val_n]
    testX = x[train_n + val_n:]
    testY = y[train_n + val_n:]

    print(f"total_n: {len(x)}, train_n: {len(trainX)}, val_n: {len(valX)}, test_n: {len(testX)}")
    return (trainX, trainY), (valX, valY), (testX, testY)

def data_process():
    n = 100000  # Reduced size for testing
    x = np.random.randint(0, n, n)
    y = np.array([my_polynomial(val) for val in x], dtype=np.float32)
    x = x.reshape(-1, 1).astype(np.float32)
    print(x[:2])
    print(y[:2])
    return x, y

def my_polynomial(x):
    return 7 * x**4 + 5 * x**3 + 2 * x**2 - 7 * x + 10

def build_model():
    inputs = Input((1,))
    h2 = Dense(8, activation='relu', name='h2')(inputs)
    h3 = Dense(16, activation='relu', name='h3')(h2)
    h4 = Dense(4, activation='relu', name='h4')(h3)
    outputs = Dense(1, name='output_layer')(h4)

    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

if __name__ == "__main__":
    main()
