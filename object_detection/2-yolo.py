#!/usr/bin/env python3
"""Task 2"""
import numpy as np
import tensorflow as tf


class Yolo:
    """
    Write a class Yolo that uses the Yolo v3 algorithm to perform
      object detection:

        class constructor: def __init__(self, model_path, classes_path,
          class_t, nms_t, anchors):
            model_path is the path to where a Darknet Keras model is stored
            classes_path is the path to where the list of class names used for
              the Darknet model, listed in order of index, can be found
            class_t is a float representing the box score threshold for the
              initial filtering step
            nms_t is a float representing the IOU threshold for non-max
              suppression
            anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
              containing all of the anchor boxes:
                outputs is the number of outputs (predictions) made by the
                  Darknet model
                anchor_boxes is the number of anchor boxes used for each
                  prediction
                2 => [anchor_box_width, anchor_box_height]
        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """init"""

        # the path to where a Darknet Keras model is stored
        self.model = tf.keras.models.load_model(model_path)

        # classes_path is the path to where the list of class names used
        #   for the Darknet model, listed in order of index, can be found
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # a float representing the box score threshold for the initial
        #   filtering step
        self.class_t = class_t

        # a float representing the IOU threshold for non-max suppression
        self.nms_t = nms_t

        # a numpy.ndarray of shape (outputs, anchor_boxes, 2) containing all of
        # the anchor boxes:
        #     outputs is the number of outputs (predictions) made by the
        #       Darknet model
        #     anchor_boxes is the number of anchor boxes used for each
        #       prediction
        #     2 => [anchor_box_width, anchor_box_height]
        self.anchors = anchors

    def sigmoid(self, arr):
        """sigmoid activation function"""
        return 1 / (1+np.exp(-1*arr))

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions from the
          Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
              anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                  used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image's original size
          [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
              anchor_boxes, 4) containing the processed boundary boxes for
              each output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                  original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
              grid_width, anchor_boxes, 1) containing the box confidences for
              each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
              grid_width, anchor_boxes, classes) containing the box's class
              probabilities for each output, respectively

        """
        # set image height and width
        image_height, image_width = tuple(image_size)

        # get input dimensions
        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value

        # create empty lists
        boxes, box_confidences, box_class_probs = [], [], []
        x_corners, y_corners = [], []

        for output in outputs:
            # set grid dimensions and achor boxes
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            # fill output lists
            boxes.append(output[..., :4])
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

            # create x-coordinate grid
            cx = np.arange(grid_width)[None, ...]
            cx = np.repeat(cx, grid_height, axis=0)

            # create y-coodinate grid
            cy = np.arange(grid_width)[None, ...]
            cy = np.repeat(cy, grid_height, axis=0).T

            # Fill corner dimensions
            x_corners.append(
                np.repeat(cx[..., np.newaxis], anchor_boxes, axis=2))
            y_corners.append(
                np.repeat(cy[..., np.newaxis], anchor_boxes, axis=2))

        for x, box in enumerate(boxes):
            bx = (self.sigmoid(box[..., 0]) + x_corners[x])/outputs[x].shape[1]
            by = (self.sigmoid(box[..., 1]) + y_corners[x])/outputs[x].shape[0]
            bw = (np.exp(box[..., 2]) * self.anchors[x, :, 0]) / input_width
            bh = (np.exp(box[..., 3]) * self.anchors[x, :, 1]) / input_height

            # Move bounding box coordinates from corner to center
            box[..., 0] = (bx - (bw * .5)) * image_width
            box[..., 1] = (by - (bh * .5)) * image_height
            box[..., 2] = (bx + (bw * .5)) * image_width
            box[..., 3] = (by + (bh * .5)) * image_height

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
          anchor_boxes, 4) containing the processed boundary boxes for each
          output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, 1) containing the processed box confidences
          for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, classes) containing the processed box class
          probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
          filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
              number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
              for each box in filtered_boxes, respectively

        """
        filtered_boxes, box_classes, box_scores = None, None, None

        for i, box in enumerate(boxes):
            score = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(score, axis=-1)
            box_score = np.max(score, axis=-1)
            filter = box_score >= self.class_t

            if filtered_boxes is None:
                filtered_boxes = box[filter]
                box_classes = box_class[filter]
                box_scores = box_score[filter]
            else:
                filtered_boxes = np.concatenate(
                    (filtered_boxes, box[filter]), axis=0)
                box_classes = np.concatenate(
                    (box_classes, box_class[filter]), axis=0)
                box_scores = np.concatenate(
                    (box_scores, box_score[filter]), axis=0)

        return filtered_boxes, box_classes, box_scores
