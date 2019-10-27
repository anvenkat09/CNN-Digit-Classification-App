import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import os.path as path
import numpy as np
import os

file_path =  path.abspath(path.join(__file__ ,"../../model_weights/ideal_weights.h5"))
batch_size = 128
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def load_trained_weights():
    model = mnist_model(input_shape = (img_rows, img_cols, 1), classes = 10)
    model.load_weights(file_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model 

#CNN used for training and predictions on the Mnist model
def mnist_model(input_shape = (28, 28, 1), classes = 12):
    X_input = Input(input_shape)

 	# 26x26x32
    X = Conv2D(32, (3,3), strides=(1,1), name='conv1')(X_input)
    X = Activation('relu')(X)

	#24x24x64    
    X = Conv2D(64, (3,3), strides=(1,1), name='conv2')(X)
    X = Activation('relu')(X)
    
    #12x12x64
    X = MaxPooling2D((2,2), strides=(2,2), name='max_pool')(X)
    
    #10x10x128
    X = Conv2D(128, (3,3), strides=(1,1), name='conv3')(X)
    X = Activation('relu')(X)
    
    #5x5x128
    X = AveragePooling2D((2,2), name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='mnist_model')

    return model

def train_model():
    model = mnist_model(input_shape = (img_rows, img_cols, 1), classes = 10)    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, checkpoint],
        validation_data=(x_test, y_test))

def test_model():
    model = load_trained_weights()
    score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

def predict(image):
    model = load_trained_weights()
    predicted_results = model.predict(image)
    return np.argmax(predicted_results)

def clear_session():
    K.clear_session()

if __name__ == '__main__':
    #The model has already been trained and the weights stored in the model_weights folder
    #train_model()
    test_model()
    
