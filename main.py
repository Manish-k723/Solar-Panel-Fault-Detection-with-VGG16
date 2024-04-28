import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from img_cap import capture_and_save_image, clean_img_folder, load_and_preprocess_image

#load dataset
train_ds = tf.data.Dataset.load('train_ds')
val_ds = tf.data.Dataset.load('val_ds')

# loaded_model = tf.keras.models.load_model('my_model')
model = keras.models.load_model("my_model1.keras")
class_names= ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered', "Not a Panel"]

#load images

output_folder = "img"
image_count = 0
while True:
    image_count = capture_and_save_image(output_folder, image_count+1)
    selected_image = os.path.join(output_folder, f"img{image_count}.jpg")
    # selected_image = os.path.join(output_folder, f"img3.jpg")
    image = load_and_preprocess_image(selected_image)
    image = tf.expand_dims(image, 0)
    predictions = model.predict(image)
    predictions = predictions[0][:6]
    score = tf.nn.softmax(predictions)
    predicted_class_index = tf.argmax(predictions, axis=-1).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(image[0].numpy())
    plt.title(f'Predicted: {class_names[predicted_class_index]} (Confidence: {score[predicted_class_index]:.2f})')
    plt.axis('off')
    plt.show()

    user_input = input("Press 'q' to quit, 'c' clean the img folder and 'enter' to continue: ")
    if user_input == 'q':
        break
    elif user_input == "c":
        clean_img_folder(output_folder)
        break
