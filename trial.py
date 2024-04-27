# from data import val_ds, train_ds, class_names
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, model_from_json

model = tf.keras.models.load_model('its_model.h5')

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model4.h5")
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# print(loaded_model.summary())

# train_ds = tf.data.Dataset.load('train_ds')
# val_ds = tf.data.Dataset.load('val_ds')

# class_name= ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# plt.figure(figsize=(25, 25))
# for images, labels in train_ds.take(1):
#     for i in range(25):
#         ax = plt.subplot(5, 5, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_name[labels[i]])
#         plt.axis("off")
# plt.show()
# print(class_name)