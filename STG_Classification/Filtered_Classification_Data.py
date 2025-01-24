import os
import json
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms import InterpolationMode

def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    return image_files

current_dir = os.path.dirname(__file__)
original_rois_dir = os.path.join(current_dir, 'Original ROIs')
original_image_files = get_image_files(original_rois_dir)

filtered_positive_files = []
filtered_negative_files = []
for file_path in original_image_files:
    file_name = os.path.basename(file_path)
    if '(' in file_name and ')' in file_name:
        content = int(file_name[file_name.index('(')+1:file_name.index(')')].split()[0])

        # Filter out files with content less than 10
        if content >= 10:
            filtered_positive_files.append(file_path)
    else:
        filtered_negative_files.append(file_path)

index = 0
labels = {}

# Define the transform
transform = Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)

# Function to save image as PNG
def save_image_as_png(image_path, new_file_path):
    image = Image.open(image_path)
    image = transform(image)
    image.save(new_file_path, format='PNG')

# Positive files
for file_path in filtered_positive_files:
    new_file_name = f"{index}.png"
    new_file_path = os.path.join(current_dir, 'Filtered_Classification_Data', new_file_name)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    save_image_as_png(file_path, new_file_path)
    labels[index] = 1

    index += 1

# Negative files
for file_path in filtered_negative_files:
    new_file_name = f"{index}.png"
    new_file_path = os.path.join(current_dir, 'Filtered_Classification_Data', new_file_name)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    save_image_as_png(file_path, new_file_path)
    labels[index] = 0

    index += 1

# Save labels to json file
labels_path = os.path.join(current_dir, 'Filtered_Classification_Data', 'labels.json')
with open(labels_path, 'w') as f:
    json.dump(labels, f)