import tensorflow as tf
from tensorflow import keras

# Params
EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.90
IMG_ROWS, IMG_COLS = 28, 28 # i/p image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1) # 1 -> Only one color channel
NB_CLASSES = 10 # Output Classes = 10 digits

# Define the LeNet Network:
class LeNet:
    # define the convnet
    @staticmethod
    def build(input_shape, classes):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        # CONV => RELU => POOL : Stage 1
        model.add(keras.layers.Conv2D(filters=20, kernel_size=(5,5),
                                      activation='relu'))
        
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        # CONV => RELU => POOL : Stage 2
        model.add(keras.layers.Convolution2D(50, (5,5), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        # Flatten => RELU Layers : Stage 3
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(500, activation='relu'))
        
        # a SOFTMAX classifier
        model.add(keras.layers.Dense(classes, activation='softmax'))
        
        return model
    

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# reshape
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# normalize
X_train, X_test = X_train / 255.0, X_test / 255.0

# cast
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)


# Initialize the model and the optimizer
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

# Comile the model
model.compile(loss='categorical_crossentropy', 
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

# Model summary
model.summary()


# use TensorBoard, princess Aurora!
callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# Fit the model
history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=VERBOSE, 
                    validation_split=VALIDATION_SPLIT,
                    callbacks=callbacks)


train_score = model.evaluate(X_train, y_train, verbose=VERBOSE)
print(f"\nTrain score: {train_score[0]}")
print(f'Train accuracy: {train_score[1]}')

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])