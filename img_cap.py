import cv2
import os
import tensorflow as tf

def capture_and_save_image(output_folder, image_count):
    cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if needed
    
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
  
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to capture frame")
        return
   
    image_name = f"img{image_count}.jpg"
   
    image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(image_path, frame)
    
    cap.release()
    
    print(f"Image saved successfully: {image_path}")
    return image_count


def clean_img_folder(output_folder):
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("Img folder cleaned.")

def load_and_preprocess_image(image_path):
    IMG_HEIGHT, IMG_WIDTH = 244, 244
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image /= 255.0  # Normalize to [0,1] range
    return image  