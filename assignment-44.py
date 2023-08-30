from keras import datasets, layers, models
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_ds = keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Ariya Rayaneh\Desktop\archive (36)",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=16,
    label_mode='binary'
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Ariya Rayaneh\Desktop\archive (36)",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=16,
    label_mode='binary'
)

class_names = ['boots', 'sneakers']

model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss,Accuracy(Train & Val')
plt.title('loss & Accuracy vs Epoch(val)')
plt.legend()
plt.savefig(r"C:\Users\Ariya Rayaneh\Desktop\archive (36)\fig1.png")


test_ds = keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Ariya Rayaneh\Desktop\archive (36)",
    image_size=(224, 224),
    batch_size=16,
    label_mode='binary'
)

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds).flatten()
y_pred = np.round(y_pred)
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(r"C:\Users\Ariya Rayaneh\Desktop\archive (36)\fig2.png")
plt.show()

# img_path = '/kaggle/input/sneakers-image-dataset-pinterest/boots/0889cfdb9c6bf454cf0b06ae2be5f7f8.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0
# prediction = model.predict(img_array)
# class_names = ['boots', 'sneakers']
# print(class_names[int(prediction[0][0])])


test_ds = keras.preprocessing.image_dataset_from_directory(
r"C:\Users\Ariya Rayaneh\Desktop\archive (36)",
    image_size=(224, 224),
    batch_size=16,
    label_mode='binary'
)
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds).flatten()
y_pred = np.round(y_pred)
accuracy = np.mean(y_pred == y_true)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='test_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('loss,Accuracy(Train & test')
plt.title('loss & Accuracy vs Epoch(test)')
plt.legend()
plt.savefig(r"C:\Users\Ariya Rayaneh\Desktop\archive (36)\fig3.png")