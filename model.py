import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

train_ds = tf.data.Dataset.load('train_ds')
val_ds = tf.data.Dataset.load('val_ds')

class_names= ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']