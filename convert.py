# convert.py — must be run in Keras 3.x
from keras.models import load_model

# Load the Keras 3.x model
model = load_model("model_data/car_damage_model.h5")

# Save as TensorFlow SavedModel format (compatible with TF/Keras 2.11)
model.export("model_data/converted_model_tf")
# or use "tf"
