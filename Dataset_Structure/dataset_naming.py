import os

# Set your folder path # Base name for renaming
#folder_path = r'D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\DataSet\client_4\dataset_4\glioma_tumor'
#base_name = "glioma_tumor_client_4"
#folder_path = r'D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\DataSet\client_4\dataset_4\meningioma_tumor'
#base_name = "meningioma_tumor_client_4"
#folder_path = r'D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\DataSet\client_4\dataset_4\no_tumor'
#base_name = "no_tumor_client_4"
#folder_path = r'D:\Masrafe\Coding\Git_Hub_code\ml_project\fedarated_brain_tumor\DataSet\client_4\dataset_4\pituitary_tumor'
#base_name = "pituitary_tumor_client_4"

# Image file extensions to look for
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

# Get all image files
image_files = [f for f in os.listdir(folder_path)
               if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in image_extensions]

# Sort to keep the order consistent
image_files.sort()

# Step 1: Rename all to temp names to avoid name conflict
for idx, filename in enumerate(image_files):
    ext = os.path.splitext(filename)[1].lower()
    temp_name = f"temp_{idx}{ext}"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, temp_name))

# Step 2: Rename from temp to final names
temp_files = [f for f in os.listdir(folder_path) if f.startswith("temp_")]
temp_files.sort()  # Ensure same order

for i, filename in enumerate(temp_files, start=1):
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{base_name}_{i}{ext}"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    print(f"Renamed: {filename} -> {new_name}")
