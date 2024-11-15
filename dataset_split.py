import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "dataset"
unorganized_dir = os.path.join(dataset_dir, "unorganized")
class_0_dir = os.path.join(unorganized_dir, "class_0")
class_1_dir = os.path.join(unorganized_dir, "class_1")

# Define output paths for train, validation, and test splits
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")

# Ensure directories exist for train, validation, and test splits
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(folder, "class_0"), exist_ok=True)
    os.makedirs(os.path.join(folder, "class_1"), exist_ok=True)

# Gather image filenames for each class
class_0_files = [f for f in os.listdir(class_0_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
class_1_files = [f for f in os.listdir(class_1_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

# Split images into train, validation, and test sets (70%-15%-15% split)
def split_and_move(files, source_dir, train_dest, val_dest, test_dest):
    train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    for file in train_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(train_dest, file))
    for file in val_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(val_dest, file))
    for file in test_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(test_dest, file))

# Perform the split and move for each class
split_and_move(class_0_files, class_0_dir, os.path.join(train_dir, "class_0"), os.path.join(val_dir, "class_0"), os.path.join(test_dir, "class_0"))
split_and_move(class_1_files, class_1_dir, os.path.join(train_dir, "class_1"), os.path.join(val_dir, "class_1"), os.path.join(test_dir, "class_1"))

print("Dataset has been split and organized into train, validation, and test folders.")
