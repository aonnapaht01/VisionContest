import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the classification model and the five specific models
classify_model = tf.keras.models.load_model("food_classify.keras")
models = {
    'Burger': tf.keras.models.load_model('food_classify_Burger.keras'),
    'Dessert': tf.keras.models.load_model('food_classify_Dessert.keras'),
    'Pizza': tf.keras.models.load_model('food_classify_Pizza.keras'),
    'Ramen': tf.keras.models.load_model('food_classify_Ramen.keras'),
    'Sushi': tf.keras.models.load_model('food_classify_Sushi.keras')
}

# Define constants
folder_path = "Test Images"
csv_path = "test.csv"
image_size_class = (128, 128)  # Size for classification
image_size_pair = (224, 224)  # Size for pair comparison
class_indicate = {0: "Burger", 1: "Dessert", 2: "Pizza", 3: "Ramen", 4: "Sushi"}

# Preprocess image for classification
def preprocess_image_class(image_path):
    img = image.load_img(f"{folder_path}/{image_path}", target_size=image_size_class)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Preprocess image for pair comparison
def preprocess_image_pair(image_path):
    img = image.load_img(f"{folder_path}/{image_path}", target_size=image_size_pair)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Classify food type and return prediction with confidence
def class_prediction(image):
    predict = classify_model.predict(image)
    predicted_class_index = np.argmax(predict, axis=-1)[0]
    confidence = np.max(predict) * 100  # Convert to percentage
    predicted_class_name = class_indicate.get(predicted_class_index, "Unknown")
    return predicted_class_name, confidence

# Predict winner using specific pair model
def pair_prediction(image1, image2, model):
    predict = model.predict([image1, image2])
    return 2 if predict > 0.5 else 1  # Return 2 if Image 2 wins, 1 if Image 1 wins

# Read the CSV file
df = pd.read_csv(csv_path)

# Initialize lists for results
predictions = []

# Process each row in the DataFrame
for _, row in df.iterrows():
    img1_path = row['Image 1']
    img2_path = row['Image 2']

    # Classify both images
    image_1_class = preprocess_image_class(img1_path)
    image_2_class = preprocess_image_class(img2_path)

    food_1, confidence_1 = class_prediction(image_1_class)
    food_2, confidence_2 = class_prediction(image_2_class)

    # Choose food type based on higher confidence
    food_type = food_1 if confidence_1 >= confidence_2 else food_2

    # Preprocess images for pair comparison
    image_1_pair = preprocess_image_pair(img1_path)
    image_2_pair = preprocess_image_pair(img2_path)

    # Get the specific model for the chosen food type
    pair_model = models.get(food_type)
    if pair_model is None:
        print(f"Error: No model found for food type {food_type}")
        continue

    # Predict the winner
    winner = pair_prediction(image_1_pair, image_2_pair, pair_model)
    predictions.append(winner)

    # Print result for this row
    print(f"Image 1: {img1_path} | Prediction 1: {food_1} {confidence_1:.2f}%")
    print(f"Image 2: {img2_path} | Prediction 2: {food_2} {confidence_2:.2f}%")
    print(f"Predicted Winner: Image {winner} ({food_type})")
    print("-" * 50)

# Update DataFrame with predictions
df["Winner"] = predictions

# Save the updated CSV
df.to_csv("test_updated.csv", index=False)

print("\nPredictions saved successfully!")
