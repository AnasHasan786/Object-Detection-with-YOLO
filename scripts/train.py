import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import glob
import sys
import warnings
import tensorflow as tf
from keras.optimizers import Adam

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.model import custom_loss, model, callbacks
from components.data_processing import utils
from components.visualizations.visualizations import plot_loss_curves


# ==========================================
# Path Management
# ==========================================
def define_paths():
    """Set up directory paths and validate their existence."""
    import errno

    current_dir = os.path.abspath(os.getcwd())

    # Retrieve validated directories from utils
    validated_dirs = utils.get_data_dir(local_data_path="data/VOC2012")

    paths = {
        "train_images": validated_dirs["train_images"], 
        "train_annotations": validated_dirs["train_annotations"],
        "val_images": validated_dirs["val_images"],  
        "val_annotations": validated_dirs["val_annotations"],  
        "model_save_path": os.path.join(
            current_dir, "outputs", "models", "model.keras"
        ),  
    }

    # Create the directory for saving the model if it doesn't exist
    model_save_dir = os.path.dirname(paths["model_save_path"])
    try:
        os.makedirs(model_save_dir, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:  
            raise PermissionError(
                f"Failed to create directory: {model_save_dir}. Check permissions."
            ) from e

    # Validate all other paths and raise errors if any required path is missing
    for key, path in paths.items():
        if key != "model_save_path" and not os.path.exists(
            path
        ):  # Exclude model_save_path from checks
            raise FileNotFoundError(f"Required path not found: {key} -> {path}")

    return paths



# ==========================================
# Prepare Datasets
# ==========================================
def prepare_datasets(paths):
    """
    Prepare TensorFlow datasets for training and validation.

    Args:
        paths (dict): A dictionary containing paths for train and validation datasets.
                      Required keys: "train_images", "train_annotations", "val_images", "val_annotations".

    Returns:
        train_dataset: A TensorFlow dataset object for training.
        val_dataset: A TensorFlow dataset object for validation.
    """
    # Collect file paths for training images and annotations
    train_image_paths = sorted(glob.glob(os.path.join(paths["train_images"], "*.jpg")))
    train_annotation_paths = sorted(
        glob.glob(os.path.join(paths["train_annotations"], "*.xml"))
    )

    # Collect file paths for validation images and annotations
    val_image_paths = sorted(glob.glob(os.path.join(paths["val_images"], "*.jpg")))
    val_annotation_paths = sorted(
        glob.glob(os.path.join(paths["val_annotations"], "*.xml"))
    )

    # Create TensorFlow datasets for training
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_image_paths, train_annotation_paths))
        .map(
            utils.get_imboxes, num_parallel_calls=tf.data.AUTOTUNE
        )  
        .map(
            utils.process_data, num_parallel_calls=tf.data.AUTOTUNE
        )  
        .map(
            utils.preprocess_augment, num_parallel_calls=tf.data.AUTOTUNE
        )  
        .batch(utils.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)  
    )

    # Create TensorFlow datasets for validation
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_image_paths, val_annotation_paths))
        .map(
            utils.get_imboxes, num_parallel_calls=tf.data.AUTOTUNE
        )  
        .map(
            utils.preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )  
        .batch(utils.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Return the datasets
    return train_dataset, val_dataset


# ==========================================
# Model Training
# ==========================================
def train_model(train_dataset, val_dataset, paths):
    """Build, compile, and train the YOLO model."""
    # Create the YOLO model
    yolo_model = model.create_model()

    # Compile the model with YOLO loss and Adam optimizer
    yolo_model.compile(
        loss=custom_loss.yolo_loss,  
        optimizer=Adam(
            learning_rate=1e-3
        ),  
    )

    # Train the model
    history = yolo_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=utils.N_EPOCHS,  
        callbacks=callbacks.get_callbacks(), 
        verbose=1, 
    )

    # Save the trained model to disk
    yolo_model.save(paths["model_save_path"])
    print(f"Model saved at: {paths['model_save_path']}")

    return history


# ==========================================
# Main Function
# ==========================================
if __name__ == "__main__":
    try:
        paths = define_paths()

        # Step 1: Prepare Datasets
        train_dataset, val_dataset = prepare_datasets(paths)

        # Step 2: Train the Model
        print("Starting model training...")
        history = train_model(train_dataset, val_dataset, paths)

        # Step 3: Plot Loss Curves
        print("Plotting loss curves...")
        plot_loss_curves(history)

        print("Training and visualization completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
