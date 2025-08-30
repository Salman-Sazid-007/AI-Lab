from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(2,), name="Input_Layer")
h1 = Dense(4, activation='relu', name="Hidden_Layer_1")(inputs)
h2 = Dense(3, activation='tanh', name="Hidden_Layer_2")(h1)
outputs = Dense(2, activation='softmax', name="Output_Layer")(h2)
model = Model(inputs=inputs, outputs=outputs, name="Modified_FCNN")
model.summary()
