import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, 
                                    MaxPooling2D, Dropout, Flatten, Dense)

class FacialExpressionModel(object):
    """A Class for Predicting the emotions using the pre-trained Model weights"""
    
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # Register custom objects for newer Keras versions
        custom_objects = {
            'Sequential': Sequential,
            'Conv2D': Conv2D,
            'BatchNormalization': BatchNormalization,
            'Activation': Activation,
            'MaxPooling2D': MaxPooling2D,
            'Dropout': Dropout,
            'Flatten': Flatten,
            'Dense': Dense
        }
        
        # Load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        
        # Load weights into the model
        self.loaded_model.load_weights(model_weights_file)
        
        # Make model predict method thread-safe
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        """Predict the Emotion using our pre-trained model and return it"""
        img = np.expand_dims(img, axis=0)  # Add batch dimension if needed
        img = img.reshape(1, 48, 48, 1)    # Ensure correct input shape
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def return_probabs(self, img):
        """Return the Probabilities of each emotion using pre-trained model"""
        img = np.expand_dims(img, axis=0)  # Add batch dimension if needed
        img = img.reshape(1, 48, 48, 1)    # Ensure correct input shape
        self.preds = self.loaded_model.predict(img)
        return self.preds[0]  # Return first (and only) prediction