import os
import random
import shutil

# Path to the source folder with all images
source_folder = r'D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset\Training\pituitary_tumor'

# Path to the destination folder for the 100 selected images
destination_folder = r'D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\Dataset\Validation\pituitary_tumor'

# Number of images to move
num_images = 100

# Allowed image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

# Get all valid image files
all_images = [f for f in os.listdir(source_folder)
              if os.path.isfile(os.path.join(source_folder, f)) and os.path.splitext(f)[1].lower() in image_extensions]

# Check if enough images are available
if len(all_images) < num_images:
    raise ValueError(f"Not enough images in the folder. Found only {len(all_images)}")

# Randomly pick images
selected_images = random.sample(all_images, num_images)

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Move the selected images
for img in selected_images:
    src_path = os.path.join(source_folder, img)
    dst_path = os.path.join(destination_folder, img)
    shutil.move(src_path, dst_path)
    print(f"Moved: {img}")

print(f"\n✅ {num_images} images moved from '{source_folder}' to '{destination_folder}'")
