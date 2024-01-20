# model.py
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))
    
    print("The model has successfully trained")
    return model, hist

def save_model(model, filename):
    model.save(filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    return keras.models.load_model(filename)
