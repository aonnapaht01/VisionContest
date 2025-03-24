import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the custom model (food_classify.keras)
model = tf.keras.models.load_model('food_classify.keras')  # Load the custom model

# Assuming that the model expects the images to be resized to 224x224
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # You might have custom preprocessing for your model. Replace it if needed.
    img_array = img_array / 255.0  # If your model expects normalized input
    return img_array

def get_image_score(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    # Assuming the output is a single prediction (class probability)
    return np.max(preds)  # Change this if your model outputs something different

def compare_food_images(image1_path, image2_path):
    score1 = get_image_score(image1_path)
    score2 = get_image_score(image2_path)
    
    print(f"First score: {score1}")
    print(f"Second score: {score2}")
    
    if score1 > score2:
        print("Winner: First")
        return "First"
    elif score2 > score1:
        print("Winner: Second")
        return "Second"
    
# Read the CSV file
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

# Apply the function to update the 'Winner' column
df = df.apply(update_winner, axis=1)

# Save the updated CSV
df.to_csv('test.csv', index=False)

print("CSV Updated 'test.csv'")
