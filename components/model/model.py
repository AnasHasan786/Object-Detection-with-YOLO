import tensorflow as tf
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    LeakyReLU,
    Reshape,
)
from keras.regularizers import l2
from components.data_processing.utils import (
    NUM_FILTERS,
    OUTPUT_DIM,
    B,
    N_CLASSES,
    H,
    W,
    SPLIT_SIZE,
)

# Load Pre-trained EfficientNetB1 as the Base Model
def create_base_model(input_shape):
    """
    Creates and configures the base model using EfficientNetB1.
    The pre-trained model uses ImageNet weights for feature extraction.

    Args:
        input_shape (tuple): Shape of the input image (H, W, 3).

    Returns:
        base_model: EfficientNetB1 model with top layers removed.
    """
    base_model = tf.keras.applications.efficientnet.EfficientNetB1(
        weights="imagenet", input_shape=input_shape, include_top=False
    )
    base_model.trainable = False
    return base_model

# Create YOLO-based Model
def create_model():
    """
    Builds the YOLO-based model using EfficientNetB1 as the backbone
    and additional custom layers for object detection.

    Returns:
        model: Compiled YOLO-based TensorFlow Sequential model.
    """
    base_model = create_base_model((H, W, 3))

    # Add Custom Layers for YOLO-style Output
    model = tf.keras.Sequential(
        [
            base_model,
            Conv2D(NUM_FILTERS,(3,3), padding = 'same',kernel_initializer='he_normal',),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.1),

            Conv2D(NUM_FILTERS,(3,3),padding = 'same',kernel_initializer='he_normal',),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.1),

            Conv2D(NUM_FILTERS,(3,3),padding = 'same',kernel_initializer='he_normal',),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.1),

            Conv2D(NUM_FILTERS,(3,3),padding = 'same',kernel_initializer='he_normal',),
            LeakyReLU(negative_slope=0.1),

            Flatten(),

            Dense(NUM_FILTERS,kernel_initializer='he_normal',),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.1),

            Dropout(0.5),
            
            Dense(SPLIT_SIZE[0] * SPLIT_SIZE[0] * OUTPUT_DIM, activation="sigmoid"),
            Reshape((SPLIT_SIZE[0], SPLIT_SIZE[0], OUTPUT_DIM)),
        ]
    )

    return model

# Model Summary
if __name__ == "__main__":
    model = create_model()
    model.summary()
