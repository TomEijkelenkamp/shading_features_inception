import os
import re
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr

# Load the Inception model
model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg')

# Function to preprocess images
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

# Initialize an empty DataFrame to store metadata
metadata = pd.DataFrame(columns=['shape', 'color_index', 'metallic_index', 'roughness_index', 'rotation_angle_index', 'light_position_index', 'file_name'])

# Regular expression to parse the filename
pattern = re.compile(r'render_(\w+)_c(\d+)_m(\d+)_r(\d+)_a(\d+)_l(\d+)')

# Directory where your images are stored
image_folder = 'C:\\Users\\tomei\\Documents\\Python\\Internship\\test_week_3\\blender_images\\generated_renders\\shading_properties_3'

# Extract features and metadata
features = []
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    match = pattern.match(img_name)
    if match:
        metadata = metadata.append({
            'shape': match.group(1),
            'color_index': int(match.group(2)),
            'metallic_index': int(match.group(3)),
            'roughness_index': int(match.group(4)),
            'rotation_angle_index': int(match.group(5)),
            'light_position_index': int(match.group(6)),
            'file_name': img_name
        }, ignore_index=True)
        
        img_array = preprocess_image(img_path)
        img_features = model.predict(img_array)
        features.append(img_features[0])
        
# Convert feature list to DataFrame
features_df = pd.DataFrame(features)

# Calculate correlation
correlations = pd.DataFrame(index=metadata.columns[:-1], columns=features_df.columns)
for meta_col in metadata.columns[:-1]:  # Exclude 'file_name'
    for feature_col in features_df.columns:
        correlation, _ = pearsonr(metadata[meta_col], features_df[feature_col])
        correlations.loc[meta_col, feature_col] = correlation

# Print correlations
print(correlations)
