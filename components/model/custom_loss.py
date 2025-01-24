import tensorflow as tf
from components.data_processing.utils import N_CLASSES, SPLIT_SIZE


def compute_iou(pred_boxes, true_boxes):
  pred_mins = pred_boxes[..., :2] - pred_boxes[..., 2:] * 0.5
  pred_maxes = pred_boxes[..., :2] + pred_boxes[..., 2:] * 0.5

  true_mins = true_boxes[..., :2] - true_boxes[..., 2:] * 0.5
  true_maxes = true_boxes[..., :2] + true_boxes[..., 2:] * 0.5

  intersect_mins = tf.maximum(pred_mins, true_mins)
  intersect_maxes = tf.minimum(pred_maxes, true_maxes)
  intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.0)

  intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
  pred_area = (pred_maxes[..., 0] - pred_mins[..., 0]) * (pred_maxes[..., 1] - pred_mins[..., 1])
  true_area = (true_maxes[..., 0] - true_mins[..., 0]) * (true_maxes[..., 1] - true_mins[..., 1])

  union_area = pred_area + true_area - intersect_area
  return tf.clip_by_value(intersect_area / (union_area + 1e-10), 0.0, 1.0)


# YOLO Loss function
@tf.keras.utils.register_keras_serializable()
def yolo_loss(y_true, y_pred):
  # Split predictions and ground truths
  pred_conf = y_pred[..., 0:1]  
  pred_boxes = y_pred[..., 1:5]  
  pred_class_probs = y_pred[..., 5:5 + N_CLASSES]  

  true_conf = y_true[..., 0:1]  
  true_boxes = y_true[..., 1:5]  
  true_class_probs = y_true[..., 5:5 + N_CLASSES]  

  # Dynamically reshape to ensure matching shapes
  pred_conf = tf.reshape(pred_conf, [tf.shape(y_pred)[0], SPLIT_SIZE[0], SPLIT_SIZE[1], 1])
  true_conf = tf.reshape(true_conf, [tf.shape(y_true)[0], SPLIT_SIZE[0], SPLIT_SIZE[1], 1])

  pred_boxes = tf.reshape(pred_boxes, [tf.shape(y_pred)[0], SPLIT_SIZE[0], SPLIT_SIZE[1], 4])
  true_boxes = tf.reshape(true_boxes, [tf.shape(y_true)[0], SPLIT_SIZE[0], SPLIT_SIZE[1], 4])

  # Compute IoU (Intersection over Union)
  iou = compute_iou(pred_boxes, true_boxes)

  # Localization loss (coordinate loss)
  localization_loss = tf.reduce_sum(
      true_conf * tf.reduce_sum(tf.square(true_boxes - pred_boxes), axis=-1, keepdims=True) # keepdims=True added here
  )

  # Confidence loss
  confidence_loss = tf.reduce_sum(tf.square(true_conf - pred_conf))

  # Class probability loss
  class_loss = tf.reduce_sum(
      true_conf * tf.reduce_sum(tf.square(true_class_probs - pred_class_probs), axis=-1, keepdims=True) # keepdims=True added here
  )

  # Combine losses
  total_loss = localization_loss + confidence_loss + class_loss
  return total_loss
