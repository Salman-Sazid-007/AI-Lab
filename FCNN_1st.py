from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

inputs = Input((1,))
outputs = Dense(1, name = 'OutputLayer')(inputs)
model = Model(inputs, outputs)
model.summary()

inputs = Input((1,))
x = Dense(1, name = 'OutputLayer')(inputs)
outputs = Activation('sigmoid', name = 'sigmoid')(x)
model = Model(inputs, outputs, name = 'FCNN_with_Activation')
model.summary()

inputs = Input((1,))
x = Dense(1, activation = 'sigmoid')(inputs)
outputs = Dense(1, name = 'OutputLayer', activation = 'sigmoid')(x)
model = Model(inputs, outputs, name = 'ShallowNN')
model.summary()

inputs = Input((1,))
x = Dense(2, activation = 'sigmoid')(inputs)
x = Dense(4, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(16, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(4, activation = 'sigmoid')(x)
outputs = Dense(1, name = 'OutputLayer', activation = 'sigmoid')(x)
model = Model(inputs, outputs, name = 'DNN')
model.summary()

plot_model(model, show_shapes = True)

inputs = Input((28,28,1))
x = Flatten()(inputs)
x = Dense(2, activation = 'sigmoid')(x)
x = Dense(4, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(16, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(4, activation = 'sigmoid')(x)
outputs = Dense(1, name = 'OutputLayer', activation = 'sigmoid')(x)
model = Model(inputs, outputs, name = 'DNN')
model.summary(show_trainable = True)

num_classes = 3
inputs = Input((28, 28, 1))
x = Flatten()(inputs)
x = Dense(2, activation = 'sigmoid')(x)
x = Dense(4, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(16, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(4, activation = 'sigmoid')(x)
outputs = Dense(num_classes, name = 'OutputLayer', activation = 'softmax')(x)
model = Model(inputs, outputs, name = 'DNN')
model.summary(show_trainable = True)

# Home Work

#   Build a deep FCNN as a 10 class classifier for RGB input images.

# Define input for RGB image (32x32x3)
inputs = Input(shape=(32, 32, 3), name='Input_RGB_Image')

# Flatten the image into a 1D vector
x = Flatten(name='Flatten_Image')(inputs)

# Deep Fully Connected Layers
x = Dense(512, activation='relu', name='Dense_1')(x)
x = Dense(256, activation='relu', name='Dense_2')(x)
x = Dense(128, activation='relu', name='Dense_3')(x)
x = Dense(64, activation='relu', name='Dense_4')(x)

# Output Layer with 10 neurons (one for each class)
outputs = Dense(10, activation='softmax', name='OutputLayer')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs, name='Deep_FCNN_RGB_10class')

# Display model architecture
model.summary()

# Optional: Visualize the model architecture (save as PNG)
plot_model(model, show_shapes=True, show_layer_names=True, to_file='fc_model.png')

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Dummy training example (if dataset available)
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
