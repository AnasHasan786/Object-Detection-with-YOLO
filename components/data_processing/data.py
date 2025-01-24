import os
import shutil
import subprocess
from utils import validate_directories

# Paths relative to the current script's directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
KAGGLE_JSON_PATH = os.path.join(
    BASE_DIR, "components", "data_processing", "kaggle.json"
)
DATASET_ZIP = os.path.join(DATA_DIR, "dataset.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "VOC2012")
VAL_IMAGES_DIR = os.path.join(EXTRACT_PATH, "ValJPEGImages")
VAL_ANNOTATIONS_DIR = os.path.join(EXTRACT_PATH, "ValAnnotations")


def setup_kaggle_environment():
    """
    Sets up the Kaggle API environment by ensuring the Kaggle JSON file exists
    and configuring the environment variable.
    """
    if os.path.exists(KAGGLE_JSON_PATH):
        os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(KAGGLE_JSON_PATH)
        print("Kaggle configuration directory set successfully.")
    else:
        raise FileNotFoundError(
            f"kaggle.json not found at {KAGGLE_JSON_PATH}. Please provide it to download the dataset."
        )


def download_and_extract_dataset(dataset_name):
    """
    Downloads the dataset from Kaggle and ensures it is extracted properly.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download dataset if not already downloaded
    if not os.path.exists(DATASET_ZIP):
        print(f"Downloading dataset {dataset_name} from Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", dataset_name, "-p", DATA_DIR, "--unzip"],
            check=True,
        )
    else:
        print(f"Dataset archive already exists: {DATASET_ZIP}")

    # Validate extraction
    if not os.path.exists(EXTRACT_PATH) or not os.listdir(EXTRACT_PATH):
        raise FileNotFoundError(
            f"Dataset not extracted properly. Please check or manually unzip {DATASET_ZIP}."
        )

    print(f"Dataset is ready at {EXTRACT_PATH}.")


def move_validation_files(data_dir_path, val_list):
    """
    Moves validation files from training directories to validation directories.
    """
    # Ensure validation directories are created before validating
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_ANNOTATIONS_DIR, exist_ok=True)

    # Validate training directories (JPEGImages and Annotations)
    dirs = validate_directories(data_dir_path)

    train_images = dirs["train_images"]
    train_annotations = dirs["train_annotations"]

    for name in val_list:
        # Move annotation files
        source_ann = os.path.join(train_annotations, name[:-3] + "xml")
        dest_ann = os.path.join(VAL_ANNOTATIONS_DIR, name[:-3] + "xml")
        if os.path.exists(source_ann):
            shutil.move(source_ann, dest_ann)
            print(f"Moved annotation: {source_ann} -> {dest_ann}")
        else:
            print(f"Annotation not found: {source_ann}")

        # Move image files
        source_img = os.path.join(train_images, name)
        dest_img = os.path.join(VAL_IMAGES_DIR, name)
        if os.path.exists(source_img):
            shutil.move(source_img, dest_img)
            print(f"Moved image: {source_img} -> {dest_img}")
        else:
            print(f"Image not found: {source_img}")


def prepare_validation_set(val_list):
    """
    Prepares the validation dataset by moving specified files from training to validation directories.
    """
    print("Preparing validation set...")
    move_validation_files(EXTRACT_PATH, val_list)


def main():
    """
    Main function to set up the dataset and prepare validation split.
    """
    # Dataset details
    dataset_name = "huanghanchina/pascal-voc-2012"

    # Validation files list
    val_list = [
        "2007_000027.jpg",
        "2007_000032.jpg",
        "2007_000033.jpg",
        "2007_000039.jpg",
        "2007_000042.jpg",
        "2007_000061.jpg",
        "2007_000063.jpg",
        "2007_000068.jpg",
        "2007_000121.jpg",
        "2007_000123.jpg",
        "2007_000129.jpg",
        "2007_000170.jpg",
        "2007_000175.jpg",
        "2007_000187.jpg",
        "2007_000241.jpg",
        "2007_000243.jpg",
        "2007_000250.jpg",
        "2007_000256.jpg",
        "2007_000272.jpg",
        "2007_000323.jpg",
        "2007_000332.jpg",
        "2007_000333.jpg",
        "2007_000346.jpg",
        "2007_000363.jpg",
        "2007_000364.jpg",
        "2007_000392.jpg",
        "2007_000423.jpg",
        "2007_000452.jpg",
        "2007_000464.jpg",
        "2007_000480.jpg",
        "2007_000491.jpg",
        "2007_000504.jpg",
        "2007_000515.jpg",
        "2007_000528.jpg",
        "2007_000529.jpg",
        "2007_000549.jpg",
        "2007_000559.jpg",
        "2007_000572.jpg",
        "2007_000584.jpg",
        "2007_000629.jpg",
        "2007_000636.jpg",
        "2007_000645.jpg",
        "2007_000648.jpg",
        "2007_000661.jpg",
        "2007_000663.jpg",
        "2007_000664.jpg",
        "2007_000676.jpg",
        "2007_000713.jpg",
        "2007_000720.jpg",
        "2007_000727.jpg",
        "2007_000733.jpg",
        "2007_000738.jpg",
        "2007_000762.jpg",
        "2007_000768.jpg",
        "2007_000783.jpg",
        "2007_000793.jpg",
        "2007_000799.jpg",
        "2007_000804.jpg",
        "2007_000807.jpg",
        "2007_000822.jpg",
        "2007_001299.jpg",
        "2007_001311.jpg",
        "2007_001321.jpg",
        "2007_001340.jpg",
    ]

    # Setup Kaggle environment
    setup_kaggle_environment()

    # Download and extract dataset
    download_and_extract_dataset(dataset_name)

    # Prepare validation split
    prepare_validation_set(val_list)


if __name__ == "__main__":
    main()
