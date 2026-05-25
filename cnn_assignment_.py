import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
from PIL import Image
from tensorflow.keras.preprocessing import image
import random

# Download dataset using keras built-in (no kaggle API needed for VS Code)
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()

# Save as CSV to match original workflow
os.makedirs("fashionmnist", exist_ok=True)
train_path = "fashionmnist/fashion-mnist_train.csv"
test_path  = "fashionmnist/fashion-mnist_test.csv"

if not os.path.exists(train_path):
    train_df = pd.DataFrame(x_train_raw.reshape(-1, 784), columns=[f"pixel{i}" for i in range(784)])
    train_df.insert(0, "label", y_train_raw)
    train_df.to_csv(train_path, index=False)

if not os.path.exists(test_path):
    test_df = pd.DataFrame(x_test_raw.reshape(-1, 784), columns=[f"pixel{i}" for i in range(784)])
    test_df.insert(0, "label", y_test_raw)
    test_df.to_csv(test_path, index=False)

test_path  = "fashionmnist/fashion-mnist_test.csv"
train_path = "fashionmnist/fashion-mnist_train.csv"

df_train = pd.read_csv(train_path)
print(df_train['label'].head())

label_names = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
print(label_names)

random_idx = np.random.randint(0, len(df_train))
image_data = df_train.iloc[random_idx]
label = image_data['label']
pixels = image_data.drop('label').values
img = pixels.reshape(28, 28)
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

print(f"The randomly selected image has label number: {label}")
print(f"The name for this label is: {label_names[label]}")

x = df_train.drop('label', axis=1)
y = df_train['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"Shape of manually split x_train: {x_train.shape}")
print(f"Shape of manually split x_test: {x_test.shape}")
print(f"Shape of manually split y_train: {y_train.shape}")
print(f"Shape of manually split y_test: {y_test.shape}")

y_train.shape
y_test.shape

# FIX: separate train and val generators (original used same generator for both)
# datagen = ImageDataGenerator(
#     rescale=1/255,
#     validation_split=0.3,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
# )
datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,

    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

x_train_reshaped = x_train.values.reshape(-1, 28, 28, 1)

# FIX: separate train_generator and val_generator
train_generator = datagen.flow(
    x_train_reshaped,
    y_train,
    batch_size=128,
    subset='training'
)

val_generator = datagen.flow(
    x_train_reshaped,
    y_train,
    batch_size=256,
    subset='validation'
)

x, y_batch = next(val_generator)
for i in range(6):
    plt.imshow(x[i])
    plt.title(f"Label: {y_batch[i]}")
    plt.axis('off')
    plt.show()

x, y_batch = next(val_generator)
for i in range(6):
    plt.imshow(x[i])
    plt.title(f"Label: {y_batch[i]}")
    plt.axis('off')
    plt.show()

x, y_batch = next(val_generator)
print(x.shape)

# model = Sequential([
#     # 1st layer CNN
#     # FIX: added activation='relu' (was missing in original)
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
#                            input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),

#     # 2nd layer CNN
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),

#     # 3rd layer CNN
#     tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),

#     # flatten layer
#     tf.keras.layers.Flatten(),
#     # fully connected layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     # output layer
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
model = Sequential([

    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# FIX: train_generator for training, val_generator for validation (were same before)
history = model.fit(train_generator, epochs=50, batch_size=256,
                    validation_data=val_generator, callbacks=[early_stop])

plt.figure(figsize=(10, 5))
plt.title("Model Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Model Accuracy")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

x_test_reshaped = x_test.values.reshape(-1, 28, 28, 1)

# FIX: test data should NOT be augmented — plain rescale only
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow(
    x_test_reshaped,
    y_test,
    batch_size=256,
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# FIX: accuracy_score needs y_pred — original had accuracy_score(y_test) which is wrong
x_test_norm = x_test_reshaped / 255.0
y_pred = np.argmax(model.predict(x_test_norm), axis=1)
print(f"Sklearn Accuracy: {accuracy_score(y_test, y_pred)}")

model.save("fashionmnist.keras")

load_model = tf.keras.models.load_model('fashionmnist.keras')
# Fashion MNIST se ek image save karo
import tensorflow as tf
import numpy as np
from PIL import Image

sample_image = x_train.iloc[0].values.reshape(28, 28)
img = Image.fromarray(sample_image.astype('uint8'))
img.save('test_image.jpg')
print("test_image.jpg saved!")

# Ab yeh line sahi ho jayegi:
# Single image prediction
image_path = 'test_image.jpg'  # Replace with your image path
img = Image.open(image_path)
img = img.resize((28, 28))
img = img.convert('L')
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.show()

"""Now that the image is prepared, you can use `model.predict()` to classify it."""

predictions = load_model.predict(img_array)
predicted_class = np.argmax(predictions[0])
predicted_label_name = label_names[predicted_class]

print(f"The predicted class for the image is: {predicted_class}")
print(f"The predicted item is: {predicted_label_name}")