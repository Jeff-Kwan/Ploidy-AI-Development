import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    return image_files

def process_image(i, annotated_file, original_files, original_image_files, segmentation_dir, pbar):
    try:
        if '(' in annotated_file and ')' in annotated_file:
            # Load the annotated image
            annotated_image = Image.open(annotated_file)
            annotated_image_np = np.array(annotated_image)
            mask = np.zeros(annotated_image_np.shape[:2], dtype=np.uint8)
            mask[(annotated_image_np[:, :, 0] < 20) & (annotated_image_np[:, :, 1] > 230) & (annotated_image_np[:, :, 2] > 230)] = 1
            mask = Image.fromarray(mask)
            mask = mask.resize((1024, 1024), resample=Image.Resampling.BICUBIC)
            mask = mask.point(lambda p: 255 if p > 0 else 0)
            mask.save(os.path.join(segmentation_dir, f"{i}_mask.png"))

            # Also save the original annotation as reference
            annotated_image = annotated_image.resize((1024, 1024), resample=Image.Resampling.BICUBIC)
            annotated_image.save(os.path.join(segmentation_dir, f"{i}_annotated.png"))

            # Find the original image
            original_index = original_files.index(os.path.splitext(os.path.basename(annotated_file))[0][:-10])
            original_image = Image.open(original_image_files[original_index])
            original_image = original_image.resize((1024, 1024), resample=Image.Resampling.BICUBIC)
            original_image.save(os.path.join(segmentation_dir, f"{i}.png"))
        else:
            pass
    finally:
        pbar.update(1)

def main():
    current_dir = os.path.dirname(__file__)
    original_rois_dir = os.path.join(current_dir, 'Original ROIs')
    annotated_rois_dir = os.path.join(current_dir, 'Annotated ROIs')
    original_image_files = get_image_files(original_rois_dir)
    annotated_image_files = get_image_files(annotated_rois_dir)

    # Assert the two lists have the same end file names (in different dirs though) (set comparison)
    assert len(original_image_files) == len(annotated_image_files)
    original_files = [os.path.splitext(os.path.basename(f))[0] for f in original_image_files]
    annotated_files = [os.path.splitext(os.path.basename(f))[0][:-10] for f in annotated_image_files]
    assert set(original_files) == set(annotated_files)
    print("Asserted one-to-one correspondence between original and annotated images")
    # Note that they may not be in order for some reason

    # New Segmentation Dir
    segmentation_dir = os.path.join(current_dir, 'Segmentation_Data')
    os.makedirs(segmentation_dir, exist_ok=True)

    # Use ThreadPoolExecutor to process images in parallel with tqdm progress bar
    with ThreadPoolExecutor() as executor, tqdm(total=len(annotated_image_files)) as pbar:
        futures = [
            executor.submit(process_image, i, annotated_file, original_files, original_image_files, segmentation_dir, pbar)
            for i, annotated_file in enumerate(annotated_image_files)
        ]
        for future in futures:
            future.result()  # Wait for all threads to complete

    print("Segmentation data created")

if __name__ == "__main__":
    main()