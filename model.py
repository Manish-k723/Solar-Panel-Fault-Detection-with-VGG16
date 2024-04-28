import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from base_model import base_model, model

train_ds = tf.data.Dataset.load('train_ds')
val_ds = tf.data.Dataset.load('val_ds')

class_names= ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epoch = 15
history = model.fit(train_ds, validation_data=val_ds, epochs=epoch,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-2,
            patience=3,
            verbose=1,
        )
    ]
)

model.save('my_model1.keras')
# keras.saving.save_model(model, 'my_model2.keras')