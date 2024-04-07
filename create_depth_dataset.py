from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from shutil import copyfile
import torch
from ultralytics import YOLO
import os
import shutil
import json
from collections import Counter
import random

def predict_image_category(image_path, processor, model, CATEGORIES, confidence_threshold=0.7):
    """Predicts the category of an image using a provided model.

    Args:
        image_path (str): Path to the image file.
        processor: The image and text preprocessing object.
        model: The image classification model.
        CATEGORIES (list): List of possible categories.
        confidence_threshold (float, optional): Minimum confidence required for a prediction.
                                             Defaults to 0.8.

    Returns:
        str: The predicted category, or "other" if the confidence is below the threshold.
    """
    try:
        image = Image.open(image_path)
        inputs = processor(text=CATEGORIES, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        max_prob, max_prob_index = torch.max(probs, dim=1)  # More efficient
        if max_prob > confidence_threshold:
            max_prob_category = CATEGORIES[max_prob_index.item()]
        else:
            max_prob_category = "other"

        return max_prob_category

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def process_images_in_directory(directory, processor, model, CATEGORIES, output_directory):
    """Processes all images in a directory and saves them into new directories based on predicted categories.

    Args:
        directory (str): Path to the directory containing image files.
        processor: The image and text preprocessing object.
        model: The image classification model.
        CATEGORIES (list): List of possible categories.
        output_directory (str): Path to the directory where images will be saved.
    """
    os.makedirs(output_directory, exist_ok=True)

    for category in CATEGORIES:
        os.makedirs(os.path.join(output_directory, category), exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            predicted_category = predict_image_category(image_path, processor, model, CATEGORIES)
            if predicted_category is not None:
                output_category_dir = os.path.join(output_directory, predicted_category)
                output_image_path = os.path.join(output_category_dir, filename)
                copyfile(image_path, output_image_path)
                print(f"save into {output_image_path}")


# Define the paths and categories
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
input_directory = "/cs/usr/evyatar613/josko_lab/downloads/depthData/train/RGB/"
output_directory = "/cs/usr/evyatar613/josko_lab/downloads/depthData/train/RGB_by_class/"
CATEGORIES = ["cat", "dog", 'horse', 'person', 'plant', 'cake', 'camera', 'flower', 'mushroom',

              'shoes', 'car', 'motorcycle', 'food', 'building', 'drink', 'bicycle', 'other', 'people',
              "children", "fish", "lizard", "fruit", 'toy', 'animal', "turtle", "bike", "statue",
              "butterfly"]

# Process images in the directory and save them into new directories
# process_images_in_directory(input_directory, processor, model, CATEGORIES, output_directory)

# print(predict_image_category("/cs/usr/evyatar613/josko_lab/downloads/depthData/train/RGB_by_class/other/COME_Train_76.jpg",processor,model,CATEGORIES))



def detect_and_save(image_dir, output_dir):
    model = YOLO('yolov8n.pt')  # Load YOLOv8 model

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, filename)
            results = model(img_path)
            try:
                result = results[0]
                box = result.boxes[0]
                print("Object type:", box.cls)
                print("Coordinates:", box.xyxy)
                print("Probability:", box.conf)
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                predicted_category = result.names[box.cls[0].item()]
                conf = round(box.conf[0].item(), 2)
                print("Object type:", predicted_category)
                print("Coordinates:", cords)
                if conf < 0.9:
                    predicted_category = "other"
                print("Probability:", conf)

                output_category_dir = os.path.join(output_dir, predicted_category)
                if not os.path.exists(output_category_dir):
                    os.makedirs(output_category_dir)
                output_image_path = os.path.join(output_category_dir, filename)

                shutil.copy2(img_path, output_image_path)
            except Exception:
                print("didnt found somthing")


def categories_dir(input_dir, unsorted_folder, folder_name):
    output_directory = f'/cs/usr/evyatar613/josko_lab/downloads/depthData/train/{folder_name}_by_class/'
    names = [i.replace(".png", "").replace(".jpg", "") for i in os.listdir(unsorted_folder)]
    for category in os.listdir(input_dir):
        curr_category = os.listdir(os.path.join(input_dir, category))
        for file in curr_category:
            output_image_path = os.path.join(output_directory, category)
            if not os.path.exists(output_image_path):
                os.makedirs(output_image_path)
            file_name = file.replace(".png", "").replace(".jpg", "")
            if file_name in names:
                shutil.copy2(os.path.join(unsorted_folder, file.replace("png", "jpg")),
                             os.path.join(output_image_path, file.replace("png", "jpg")))
                # shutil.copy2( os.path.join(unsorted_folder,file), os.path.join(output_image_path,file))




import os
import json
import random
from sklearn.model_selection import train_test_split

def create_json_data(data_folder, min_examples, output_folder, train_split=0.6, val_split=0.2):
    """Creates JSON files with distinct classes for few-shot train, val, and test sets.

    Args:
        data_folder: Path to the folder containing the image classes.
        min_examples: Minimum number of examples required for a class.
        output_folder: Path to the folder where JSON files will be saved.
        train_split:  Percentage of class folders to use for the train set.
        val_split: Percentage of class folders to use for the val set.
    """

    all_class_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

    num_classes = len(all_class_folders)
    num_train_classes = int(num_classes * train_split)
    num_val_classes = int(num_classes * val_split)
    num_test_classes = num_classes - num_train_classes - num_val_classes

    # Split class folders into train, val, test
    random.shuffle(all_class_folders)  # Shuffle for random assignment
    train_classes = all_class_folders[:num_train_classes]
    val_classes = all_class_folders[num_train_classes:num_train_classes + num_val_classes]
    test_classes = all_class_folders[num_train_classes + num_val_classes:]

    # Build datasets for each split
    datasets = {}
    label_index = 0
    for split, class_folders in zip(["train", "val", "test"], [train_classes, val_classes, test_classes]):
        label_names = []
        image_names = []
        image_labels = []

        for i, class_folder in enumerate(class_folders):
            class_path = os.path.join(data_folder, class_folder)
            images_in_class = [
                os.path.join(class_path, img) for img in os.listdir(class_path)
                if img.endswith(".jpg") or img.endswith(".png")
            ]

            if len(images_in_class) >= min_examples:
                label_names.append(class_folder)
                image_names.extend(images_in_class)
                image_labels.extend([label_index] * len(images_in_class))
                label_index += 1

        datasets[split] = {
            "label_names": label_names,
            "image_names": image_names,
            "image_labels": image_labels
        }

    # Save JSON files
    os.makedirs(output_folder, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_folder, f"{split}.json"), "w") as f:
            json.dump(datasets[split], f)


def select_and_move_images(data_folder, min_examples, output_dir, num_images):
    """Selects random images from each class and moves them to the output directory.

    Args:
        data_folder: Path to the folder containing the image classes.
        min_examples: Minimum number of examples required for a class.
        output_dir: Path to the output directory (RGB_100).
        num_images: Number of images to select per class.
    """

    os.makedirs(output_dir, exist_ok=True)  # Create the output directory

    for class_folder in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_folder)
        if os.path.isdir(class_path):
            images_in_class = [
                os.path.join(class_path, img)
                for img in os.listdir(class_path)
                if img.endswith(".jpg") or img.endswith(".png")
            ]

            if len(images_in_class) >= min_examples:
                # Randomly select images
                selected_images = random.sample(images_in_class, min(num_images, len(images_in_class)))

                # Create class folder in the output directory
                output_class_dir = os.path.join(output_dir, class_folder)
                os.makedirs(output_class_dir, exist_ok=True)

                # Move selected images
                for image_path in selected_images:
                    shutil.copy2(image_path, output_class_dir)

# if __name__ == "__main__":
#     DATA_FOLDER = "/cs/usr/evyatar613/Desktop/josko_lab/downloads/depthData/class_dataset/images/RGB"
#     MIN_EXAMPLES_PER_CLASS = 100
#     RGB_100_DIR = "/cs/usr/evyatar613/Desktop/josko_lab/downloads/depthData/class_dataset/images/RGB_100"
#     NUM_IMAGES_PER_CLASS = 100
#     select_and_move_images(DATA_FOLDER, MIN_EXAMPLES_PER_CLASS, RGB_100_DIR, NUM_IMAGES_PER_CLASS)
#
#



if __name__ == "__main__":
    DATA_FOLDER = "/cs/usr/evyatar613/Desktop/josko_lab/downloads/depthData/class_dataset/images/RGB_100"
    MIN_EXAMPLES_PER_CLASS = 100
    OUTPUT_FOLDER ="/cs/usr/evyatar613/Desktop/josko_lab/downloads/depthData/class_dataset/json_files"
    create_json_data(DATA_FOLDER, MIN_EXAMPLES_PER_CLASS, OUTPUT_FOLDER)





# for dest in ["DES","DUT-RGBD","LFSD","NJU2K","NLPR","ReDWeb-S","SIP","STERE"]:
#     image_dir = f'/cs/usr/evyatar613/josko_lab/downloads/depthData/test/{dest}/RGB/'
#     output_dir = f'/cs/usr/evyatar613/josko_lab/downloads/depthData/test/{dest}/RGB_by_class/'
#     detect_and_save(image_dir, output_dir)
#
#
#     image_dir = output_dir+'other'
#
#     output_dir = output_dir+'/clip/'
#
#     process_images_in_directory(image_dir,processor,model,CATEGORIES,output_dir)
#     print(f"finish {dest}")
#
# # input_dir = "/cs/usr/evyatar613/josko_lab/downloads/depthData/train/depths_by_class"
# # output_dir = "/cs/usr/evyatar613/josko_lab/downloads/depthData/train/RGB"
# # #
# # categories_dir(input_dir, output_dir, "RGB")
