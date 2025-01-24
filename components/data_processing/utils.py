import os
import shutil
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import albumentations as A
import cv2

# ==========================================
# Global Constants
# ==========================================
H, W = 224, 224
SPLIT_SIZE = (7, 7)
B = 2
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
N_CLASSES = 20
BATCH_SIZE = 32
OUTPUT_DIM = 30
N_EPOCHS = 100
NUM_FILTERS = 512

# Define the base directory explicitly (go up from the components folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the outputs folder outside the components folder
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)  # Ensure the outputs folder exists

# Define the callbacks folder inside the correct outputs folder
CALLBACKS_DIR = os.path.join(OUTPUTS_DIR, "callbacks")
os.makedirs(CALLBACKS_DIR, exist_ok=True)  # Ensure the callbacks folder exists

# ==========================================
# Default Paths
# ==========================================
DEFAULT_LOCAL_PATH = os.path.join("data", "VOC2012")

# ==========================================
# Data Directory Setup
# ==========================================
def get_data_dir(local_data_path=DEFAULT_LOCAL_PATH):
    """
    Sets up the data directory for local runs.
    """
    try:
        print("Using local data path.")
        data_dir_path = local_data_path

        # Call the validated_dirs function with the base directory
        validated_directories = validated_dirs(data_dir_path)
        print(f"Data directory set to: {data_dir_path}")
        return validated_directories
    except Exception as e:
        print(f"Error setting up data directory: {e}")
        raise RuntimeError(
            "Failed to set up the data directory. Check your configuration or paths."
        ) from e

# ==========================================
# Validate Directories
# ==========================================
def validated_dirs(base_dir):
    """
    Validates the required directories in the dataset.

    Parameters:
        base_dir (str): Path to the base directory containing the dataset.

    Returns:
        dict: Dictionary containing paths to the validated directories.

    Raises:
        FileNotFoundError: If any of the required directories are missing.
    """
    required_dirs = {
        "train_images": "JPEGImages",
        "train_annotations": "Annotations",
        "val_images": "ValJPEGImages",
        "val_annotations": "ValAnnotations",
    }
    validated_paths = {}

    for key, sub_dir in required_dirs.items():
        dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required directory '{sub_dir}' not found in {base_dir}")
        validated_paths[key] = dir_path

    print("All required directories are validated.")
    return validated_paths

# ==========================================
# Preprocessing xml
# ==========================================
def preprocess_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    size_tree = root.find("size")
    height = float(size_tree.find("height").text)
    width = float(size_tree.find("width").text)

    bounding_boxes = []

    for object_tree in root.findall("object"):
        for bounding_box in object_tree.iter("bndbox"):
            xmin = float(bounding_box.find("xmin").text)
            ymin = float(bounding_box.find("ymin").text)
            xmax = float(bounding_box.find("xmax").text)
            ymax = float(bounding_box.find("ymax").text)
            break

        class_name = object_tree.find("name").text
        class_dict = {CLASSES[i]: i for i in range(len(CLASSES))}

        bounding_box = [
            (xmin + xmax) / (2 * width),
            (ymin + ymax) / (2 * height),
            (xmax - xmin) / width,
            (ymax - ymin) / height,
            class_dict[class_name],
        ]

        bounding_boxes.append(bounding_box)

    return tf.convert_to_tensor(bounding_boxes, dtype=tf.float32)

# ==========================================
# Data Transformation Utilities
# ==========================================
def generate_output(bounding_boxes):
    """
    Generates output labels for a given set of bounding boxes.

    Parameters:
        bounding_boxes (tf.Tensor): Bounding box data.

    Returns:
        tf.Tensor: Output label tensor.
    """
    output_label = np.zeros((SPLIT_SIZE[0], SPLIT_SIZE[1], OUTPUT_DIM))
    for b in range(len(bounding_boxes)):
        grid_x = bounding_boxes[b, 0] * SPLIT_SIZE[0]
        grid_y = bounding_boxes[b, 1] * SPLIT_SIZE[1]
        i, j = int(grid_x), int(grid_y)

        output_label[i, j, 0:5] = [
            1.0,
            grid_x % 1,
            grid_y % 1,
            bounding_boxes[b, 2],
            bounding_boxes[b, 3],
        ]
        output_label[i, j, 5 + int(bounding_boxes[b, 4])] = 1.0

    return tf.convert_to_tensor(output_label, dtype=tf.float32)

# ==========================================
# Data Augmentation Utilities
# ==========================================
transforms = A.Compose(
    [
        A.Resize(H, W),
        A.RandomCrop(
            width=np.random.randint(int(0.8 * W), W),
            height=np.random.randint(int(0.8 * H), H),
            p=0.5,
        ),
        A.RandomScale(scale_limit=0.2, interpolation=cv2.INTER_LANCZOS4, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Resize(H, W),
    ],
    bbox_params=A.BboxParams(format="yolo"),
)

def aug_albument(image, bboxes):
    """
    Applies Albumentations transformations to an image and bounding boxes.

    Parameters:
        image (np.ndarray): Input image.
        bboxes (list): Bounding boxes in YOLO format.

    Returns:
        list: Transformed image and bounding boxes as tensors.
    """
    augmented = transforms(image=image, bboxes=bboxes)
    return [
        tf.convert_to_tensor(augmented["image"], dtype=tf.float32),
        tf.convert_to_tensor(augmented["bboxes"], dtype=tf.float32),
    ]

def process_data(image, bboxes):
    """
    Processes image and bounding boxes for training.

    Parameters:
        image (tf.Tensor): Input image.
        bboxes (tf.Tensor): Bounding boxes.

    Returns:
        tuple: Processed image and bounding boxes.
    """
    aug = tf.numpy_function(
        func=aug_albument, inp=[image, bboxes], Tout=(tf.float32, tf.float32)
    )
    aug[0].set_shape([H, W, 3])
    aug[1].set_shape([None, 5])
    return aug[0], aug[1]

def preprocess_augment(img, y):
    """
    Applies augmentation and preprocessing to an image and its labels.

    Parameters:
        img (tf.Tensor): Input image.
        y (tf.Tensor): Bounding boxes.

    Returns:
        tuple: Augmented image and labels.
    """
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0, 255)
    img.set_shape([H, W, 3])

    labels = tf.numpy_function(func=generate_output, inp=[y], Tout=tf.float32)
    labels.set_shape([SPLIT_SIZE[0], SPLIT_SIZE[1], OUTPUT_DIM])
    return img, labels

def preprocess(img, y):
    """
    Preprocesses image and labels without augmentation.

    Parameters:
        img (tf.Tensor): Input image.
        y (tf.Tensor): Bounding boxes.

    Returns:
        tuple: Preprocessed image and labels.
    """
    img = tf.cast(tf.image.resize(img, size=[H, W]), dtype=tf.float32)

    labels = tf.numpy_function(func=generate_output, inp=[y], Tout=tf.float32)
    labels.set_shape([SPLIT_SIZE[0], SPLIT_SIZE[1], OUTPUT_DIM])
    return img, labels

# ==========================================
# Dataset Utilities
# ==========================================
def get_imboxes(im_path, xml_path):
    img = tf.io.decode_jpeg(tf.io.read_file(im_path))
    img = tf.cast(tf.image.resize(img, size=[H, W]), dtype=tf.float32)

    boxes = tf.numpy_function(func=preprocess_xml, inp=[xml_path], Tout=tf.float32)
    boxes.set_shape([None, 5])
    return img, boxes
