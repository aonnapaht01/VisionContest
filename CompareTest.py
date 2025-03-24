import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the five models
models = {
    'Burger': tf.keras.models.load_model('food_classify_Burger.keras'),
    'Dessert': tf.keras.models.load_model('food_classify_Dessert.keras'),
    'Pizza': tf.keras.models.load_model('food_classify_Pizza.keras'),
    'Ramen': tf.keras.models.load_model('food_classify_Ramen.keras'),
    'Sushi': tf.keras.models.load_model('food_classify_Sushi.keras')
}

# Assuming that the model expects the images to be resized to 224x224
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # You might have custom preprocessing for your model. Replace it if needed.
    img_array = img_array / 255.0  # If your model expects normalized input
    return img_array

def get_image_score(img_path, model):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    # Assuming the output is a single prediction (class probability)
    return np.max(preds)  # Change this if your model outputs something different

def compare_food_images(image1_path, image2_path, model1, model2):
    score1 = get_image_score(image1_path, model1)
    score2 = get_image_score(image2_path, model2)
    
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
    food_type1 = row['Food Type 1']  # Column that indicates the food type for image 1
    food_type2 = row['Food Type 2']  # Column that indicates the food type for image 2
    
    # Get the appropriate model for each food type
    model1 = models.get(food_type1)
    model2 = models.get(food_type2)
    
    if model1 is None or model2 is None:
        print(f"Error: Model for food type not found.")
        return row
    
    result = compare_food_images(image1_path, image2_path, model1, model2)
    
    if result == "First":
        row['Winner'] = 1
    elif result == "Second":
        row['Winner'] = 2
    
    return row

# Apply the function to update the 'Winner' column
df = df.apply(update_winner, axis=1)

# Save the updated CSV
df.to_csv('test.csv', index=False)
