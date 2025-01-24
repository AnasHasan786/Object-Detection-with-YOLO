import os
import tensorflow as tf
from components.data_processing.utils import CALLBACKS_DIR

# Get the base directory of the project (up one level from the components folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Outputs directory (outside the components folder)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)  # Ensure the outputs folder exists

# Callbacks directory inside the outputs folder
CALLBACKS_DIR = os.path.join(OUTPUTS_DIR, "callbacks")
os.makedirs(CALLBACKS_DIR, exist_ok=True)  # Ensure the callbacks folder exists

# Models directory inside the outputs folder
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure the models folder exists

# Define a global checkpoint filepath in the models folder
checkpoint_filepath = os.path.join(MODELS_DIR, "model.keras")


# Model Checkpoint Callback
def create_checkpoint_callback():
    """Create a ModelCheckpoint callback."""
    os.makedirs(CALLBACKS_DIR, exist_ok=True)  # Ensure the save directory exists
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

# Learning Rate Scheduler
def scheduler(epoch, lr):
    """
    Adjust the learning rate based on the epoch number.
    - 0 to 39 epochs: learning rate = 1e-3
    - 40 to 79 epochs: learning rate = 5e-4
    - 80+ epochs: learning rate = 1e-4
    """
    if epoch < 40:
        return 1e-3
    elif 40 <= epoch < 80:
        return 5e-4
    else:
        return 1e-4


def create_lr_scheduler():
    """Create a LearningRateScheduler callback."""
    return tf.keras.callbacks.LearningRateScheduler(scheduler)


# Combine All Callbacks
def get_callbacks():
    """
    Returns the list of callbacks for use in training.
    Returns:
        list: A list of Keras callbacks.
    """
    
    checkpoint_callback = create_checkpoint_callback()
    lr_scheduler = create_lr_scheduler()

    return [checkpoint_callback, lr_scheduler]
