

import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


base_path = "/Users/rohansujith/Desktop/Python/autofix_gpt/docs/Damage_Dataset"
train_csv_path = os.path.join(base_path, "train.csv")
image_folder_path = os.path.join(base_path, "train", "train")  


df = pd.read_csv(train_csv_path)


df['filename'] = df['filename'].str.strip()


df['label'] = df['label'].astype(int)


df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_folder_path, x))


missing_files = df[~df['filepath'].apply(os.path.exists)]
if not missing_files.empty:
    print("❌ Missing image files:")
    print(missing_files[['filename', 'filepath']])
    raise FileNotFoundError("Some image files are missing. Please fix the dataset path.")
else:
    print("✅ All image files found.")


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6

def process_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, tf.one_hot(label, depth=NUM_CLASSES)

def get_dataset(dataframe):
    filepaths = dataframe['filepath'].values
    labels = dataframe['label'].values
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = get_dataset(train_df)
val_ds = get_dataset(val_df)
print("Dataset loaded successfully.")


model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)
print(" Model trained successfully.")



# ✅ After training, save the model to disk
model_save_path = 'car_damage_model.h5'
model.save(model_save_path)
print(f"✅ Model saved to {model_save_path}")



