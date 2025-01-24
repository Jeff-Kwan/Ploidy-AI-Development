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

filtered_positive_files = []
filtered_negative_files = []
for file_path in original_jpeg_files:
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

# Positive files
for file_path in filtered_positive_files:
    file_name = os.path.basename(file_path)

    # copy the file to new directory with new name
    new_file_name = f"{index}.jpeg"
    new_file_path = os.path.join(current_dir, 'Filtered_Classification_Data', new_file_name)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    with open(file_path, 'rb') as fsrc, open(new_file_path, 'wb') as fdst:
        fdst.write(fsrc.read())
    labels[index] = 1

    index += 1

# Negative files
for file_path in filtered_negative_files:
    file_name = os.path.basename(file_path)

    # copy the file to new directory with new name
    new_file_name = f"{index}.jpeg"
    new_file_path = os.path.join(current_dir, 'Filtered_Classification_Data', new_file_name)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    with open(file_path, 'rb') as fsrc, open(new_file_path, 'wb') as fdst:
        fdst.write(fsrc.read())
    labels[index] = 0

    index += 1

# Save labels to json file
labels_path = os.path.join(current_dir, 'Filtered_Classification_Data', 'labels.json')
with open(labels_path, 'w') as f:
    json.dump(labels, f)