import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import components.data_processing.utils as utils
from components.model.callbacks import checkpoint_filepath
from components.model.custom_loss import yolo_loss


# Load the trained model
model = load_model(
    checkpoint_filepath,
    custom_objects={"yolo_loss": yolo_loss}
)

# Define directories
current_dir = os.getcwd()
outputs_directory = os.path.join(current_dir, "outputs")

images_directory = os.path.join(outputs_directory, "images")
COCO_PATH = os.path.join(images_directory, "coco_images")
os.makedirs(COCO_PATH, exist_ok=True)

# Create the 'outputs' directory for detection results
os.makedirs(outputs_directory, exist_ok=True)

# Create the 'detected_images' directory for saving detection results
detected_images_directory = os.path.join(outputs_directory, "detected_images")
os.makedirs(detected_images_directory, exist_ok=True)

def model_test(filename):
    """
    Test the model on a given image file, draw bounding boxes for detected objects,
    and save the output image with detections.

    Args:
        filename (str): Name of the image file in the 'coco_images' directory.
    """
    try:
        test_path = os.path.join(COCO_PATH, filename)
        print(f"Processing file: {test_path}")

        img = cv2.resize(cv2.imread(test_path), (utils.H, utils.W))
        image = tf.io.decode_jpeg(tf.io.read_file(test_path))
        image = tf.image.resize(image, [utils.H, utils.W])

        output = model.predict(np.expand_dims(image, axis=0))
        THRESH = 0.25

        object_positions = tf.concat(
            [tf.where(output[..., 0] >= THRESH), tf.where(output[..., 5] >= THRESH)],
            axis=0,
        )
        print("Object positions:", object_positions)

        selected_output = tf.gather_nd(output, object_positions)
        final_boxes = []
        final_scores = []

        for i, pos in enumerate(object_positions):
            for j in range(2):
                if selected_output[i][j * 5] > THRESH:
                    output_box = tf.cast(
                        output[pos[0]][pos[1]][pos[2]][(j * 5) + 1 : (j * 5) + 5],
                        dtype=tf.float32,
                    )

                    x_centre = (tf.cast(pos[1], dtype=tf.float32) + output_box[0]) * 32
                    y_centre = (tf.cast(pos[2], dtype=tf.float32) + output_box[1]) * 32
                    x_width, y_height = utils.H * output_box[2], utils.W * output_box[3]

                    x_min = max(0, int(x_centre - (x_width / 2)))
                    y_min = max(0, int(y_centre - (y_height / 2)))
                    x_max = min(utils.W, int(x_centre + (x_width / 2)))
                    y_max = min(utils.H, int(y_centre + (y_height / 2)))

                    final_boxes.append(
                        [
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                            str(
                                utils.CLASSES[
                                    tf.argmax(selected_output[..., 10:], axis=-1)[i]
                                ]
                            ),
                        ]
                    )
                    final_scores.append(selected_output[i][j * 5])

        print("Final scores:", final_scores)
        print("Final boxes:", final_boxes)

        final_boxes = np.array(final_boxes)

        nms_boxes = final_boxes[..., 0:4]
        nms_output = tf.image.non_max_suppression(
            nms_boxes,
            final_scores,
            max_output_size=100,
            iou_threshold=0.2,
            score_threshold=float("-inf"),
        )
        print("NMS output:", nms_output)

        for i in nms_output:
            cv2.rectangle(
                img,
                (int(final_boxes[i][0]), int(final_boxes[i][1])),
                (int(final_boxes[i][2]), int(final_boxes[i][3])),
                (0, 0, 255),
                1,
            )
            cv2.putText(
                img,
                final_boxes[i][-1],
                (int(final_boxes[i][0]), int(final_boxes[i][1]) + 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (2, 225, 155),
                1,
            )

        output_path = os.path.join(detected_images_directory, filename[:-4] + "_det.jpg")
        cv2.imwrite(output_path, cv2.resize(np.array(img), (384, 384)))
        print(f"Saved detection image to {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    test_filename = "000000000081.jpg"
    model_test(test_filename)
