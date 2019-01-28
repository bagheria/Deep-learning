# Neural network workflow by Francois Chollet, in his book "Deep Learning with Python"
# 1. Define the training data
# 2. Define a neural network model
# 3. Configure the learning process
# 4. Train the model

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics

# 1. Define the training data:
X_train = np.random.random((5000, 32))
y_train = np.random.random((5000, 5))


# 2. Define the neural network model:
# The Sequential class is used to define a linear stack of network layers which then, collectively, constitute a model.
# We will use the Sequential constructor to create a model,
# which will then have layers added to it using the add() method.
# For multi-class classification problem, the activation function for output layer is set to softmax.
INPUT_DIM = X_train.shape[1]
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_dim=INPUT_DIM))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.summary()


# 3. Configure the learning process:
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit(X_train, y_train,
          batch_size=128,
          epochs=100)
