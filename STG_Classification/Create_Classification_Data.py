import os
import json

def get_jpeg_files(directory):
    jpeg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.jpg'):
                jpeg_files.append(os.path.join(root, file))
    return jpeg_files

current_dir = os.path.dirname(__file__)
original_rois_dir = os.path.join(current_dir, 'Original ROIs')
original_jpeg_files = get_jpeg_files(original_rois_dir)

labels = {}
for i, file_path in enumerate(original_jpeg_files):
    file_name = os.path.basename(file_path)

    # copy the file to new directory with new name
    new_file_name = f"{i}.jpeg"
    new_file_path = os.path.join(current_dir, 'Classification_Data', new_file_name)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    with open(file_path, 'rb') as fsrc, open(new_file_path, 'wb') as fdst:
        fdst.write(fsrc.read())

    # Create labels for classification () determining existence of object
    if '(' in file_name and ')' in file_name:
        labels[i] = 1
    else:
        labels[i] = 0

# Save labels to json file
labels_path = os.path.join(current_dir, 'Classification_Data', 'labels.json')
with open(labels_path, 'w') as f:
    json.dump(labels, f)