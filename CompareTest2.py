import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

model = VGG16(weights='imagenet')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_image_score(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    return np.max(preds)

def compare_food_images(image1_path, image2_path):
    score1 = get_image_score(image1_path)
    score2 = get_image_score(image2_path)
    
    print(f"First_score: {score1}")
    print(f"Second_score: {score2}")
    
    if score1 > score2:
        return "First"
    elif score2 > score1:
        return "Second"

df = pd.read_csv('test.csv')  

def update_winner(row):
    image1_path = row['Image 1']  
    image2_path = row['Image 2']  
    
    result = compare_food_images(image1_path, image2_path)
    
    if result == "First":
        row['Winner'] = 1
    elif result == "Second":
        row['Winner'] = 2
    
    return row

df = df.apply(update_winner, axis=1)

df.to_csv('test_updated.csv', index=False)