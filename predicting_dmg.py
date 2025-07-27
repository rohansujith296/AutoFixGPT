import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.models import load_model

model =  keras.models.load_model('/Users/rohansujith/Desktop/Python/autofix_gpt/car_damage_model.h5')
print("model loaded succesfully")


def classify_damage(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    preds = model.predict(img_tensor)
    class_idx = np.argmax(preds)
    
    label_map = {
        0: "Front Bumper",
        1: "Rear Bumper",
        2: "Left Door",
        3: "Right Door",
        4: "Windshield",
        5: "Hood"
    }
    return label_map[class_idx]
