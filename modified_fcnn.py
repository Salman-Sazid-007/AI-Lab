
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Step 1: Define input layer (shape = 2 features)
inputs = Input(shape=(20000,), name="Input_Layer")

# Step 2: First hidden layer with 4 neurons and ReLU activation
h1 = Dense(40000, activation='relu', name="Hidden_Layer_1")(inputs)

# Step 3: Second hidden layer with 3 neurons and tanh activation
h2 = Dense(30000, activation='tanh', name="Hidden_Layer_2")(h1)

# Step 4: Output layer with 2 neurons (for 2-class classification) using softmax
outputs = Dense(20000, activation='softmax', name="Output_Layer")(h2)

# Step 5: Build the model
model = Model(inputs=inputs, outputs=outputs, name="Modified_FCNN")

# Step 6: Show model architecture
model.summary()

# Step 7: Optional - Save visualization of the model architecture
plot_model(model, show_shapes=True, show_layer_names=True, to_file='modified_fcnn.png')
