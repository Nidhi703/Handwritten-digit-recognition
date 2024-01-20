# recognition.py
import numpy as np

def predict_digit(model, img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert RGB to grayscale
    img = img.convert('L')
    
    # Convert image to NumPy array
    img_array = np.array(img)
    
    # Reshape to support the model input and normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0
    
    # Predict the class
    prediction = model.predict([img_array])[0]
    
    return np.argmax(prediction), max(prediction)
