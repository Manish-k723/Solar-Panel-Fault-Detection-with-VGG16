import tensorflow as tf

img_height = 244
img_width = 244
train_ds = tf.keras.utils.image_dataset_from_directory(
  'Faulty_solar_panel',
  validation_split=0.2,
  subset='training',
  image_size=(img_height, img_width),
  batch_size=32,
  seed=42,
  shuffle=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'Faulty_solar_panel',
  validation_split=0.2,
  subset='validation',
  image_size=(img_height, img_width),
  batch_size=32,
  seed=42,
  shuffle=True)

tf.data.experimental.save(train_ds, 'train_ds')
tf.data.experimental.save(val_ds, 'val_ds')
