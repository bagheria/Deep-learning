from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import regularizers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def data_summary(X_train, y_train, X_test, y_test):
    """Summarize current state of dataset"""
    print('Train images shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)
    print('Train labels:', y_train)
    print('Test labels:', y_test)

def plot_data(dataindex):
    # Create a grid of 3x3 images
    for i in range(dataindex, dataindex + 9):
        plt.subplot(330 + 1 + i % 9)
        plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # Show the plot
    plt.show()

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Check state of dataset
data_summary(X_train, y_train, X_test, y_test)

# Plot couple of images
plot_data(12)

# Params
NUM_ROWS = 28
NUM_COLS = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10

# Data augmentation
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
# from keras.preprocessing.image import ImageDataGenerator
# # reshape to be [samples][pixels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# # convert from int to float
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# # define data preparation
# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# # fit parameters from data
# datagen.fit(X_train)


# Reshape data
X_train = X_train.reshape((X_train.shape[0], NUM_ROWS * NUM_COLS))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], NUM_ROWS * NUM_COLS))
X_test = X_test.astype('float32') / 255

# Categorically encode labels
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Check state of dataset
# data_summary(X_train, y_train, X_test, y_test)

model = [models.Sequential() for i in range(4)]
names = ['Fully connected neural network',
         'NN with L1 regulizer',
         'NN with L2 regulizer',
         'NN with Dropout']

# Build a fully connected neural network
# model[0] = models.Sequential()
model[0].add(Dense(512, activation='relu', input_shape=(NUM_ROWS * NUM_COLS,)))
model[0].add(Dense(256, activation='relu'))
model[0].add(Dense(10, activation='softmax'))


# Build a fully connected neural network with l1 regulizer
# model[1] = models.Sequential()
model[1].add(Dense(512, activation='relu', input_shape=(NUM_ROWS * NUM_COLS,),
                   kernel_regularizer=regularizers.l1(0.01)))
model[1].add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
model[1].add(Dense(10, activation='softmax'))


# Build a fully connected neural network with l2 regulizer
# model[2] = models.Sequential()
model[2].add(Dense(512, activation='relu', input_shape=(NUM_ROWS * NUM_COLS,),
                   kernel_regularizer=regularizers.l2(0.01)))
model[2].add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model[2].add(Dense(10, activation='softmax'))


# Build a fully connected neural network with Dropout
# model[3] = models.Sequential()
model[3].add(Dense(512, activation='relu', input_shape=(NUM_ROWS * NUM_COLS,)))
model[3].add(Dropout(0.5))
model[3].add(Dense(256, activation='relu'))
model[3].add(Dropout(0.25))
model[3].add(Dense(10, activation='softmax'))

score = []
for i in range(0, 4):
    # Compile model
    model[i].compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model[i].fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              # callbacks=[plot_losses],
              verbose=1,
              validation_data=(X_test, y_test),
              # callbacks=[EarlyStopping(monitor='val_acc', patience=2)]
              )

    score.append(model[i].evaluate(X_test, y_test, verbose=0))

for i in range(0, 4):
    print('Result for ', names[i])
    print('Test loss:', score[i][0])
    print('Test accuracy:', score[i][1])

# Summary of neural network
# model.summary()

# model.save("mnist-model.h3reg")

# Test model for one image
# img = X_test[122]
# test_img = img.reshape((1,784))
# img_class = model.predict_classes(test_img)
# prediction = img_class[0]
# classname = img_class[0]
# print("Class: ",classname)
# img = img.reshape((28,28))
# plt.imshow(img)
# plt.title(classname)
# plt.show()
