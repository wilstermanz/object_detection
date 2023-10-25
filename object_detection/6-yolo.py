#!/usr/bin/env python3
"""Task 5"""
import cv2
import numpy as np
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


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

    @staticmethod
    def _iou(b1, b2):
        """calculates intersection over union"""
        b1x1, b1y1, b1x2, b1y2 = tuple(b1)
        b2x1, b2y1, b2x2, b2y2 = tuple(b2)

        x1 = max(b1x1, b2x1)
        y1 = max(b1y1, b2y1)
        x2 = min(b1x2, b2x2)
        y2 = min(b1y2, b2y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        b1_area = (b1x2 - b1x1) * (b1y2 - b1y1)
        b2_area = (b2x2 - b2x1) * (b2y2 - b2y1)
        union = b1_area + b2_area - intersection

        return intersection / union  # intersection over union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
          filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
          for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
          each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes,
          predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
              the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing the
              class number for box_predictions ordered by class and box score,
              respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing the
              box scores for box_predictions ordered by class and box score,
              respectively
        """
        # create output lists
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # find number of unique classes
        unique_classes = np.unique(box_classes)

        # for each class
        for current_class in unique_classes:

            # create a list of the indexes of boxes for the current class
            box_indexes = np.where(box_classes == current_class)

            # Make a list of boxes and scores for the current class
            class_boxes = filtered_boxes[box_indexes]
            class_box_scores = box_scores[box_indexes]

            # while boxes remain in the class_boxes list
            while len(class_boxes > 0):

                # find the index of highest scoring box for the class
                max_score_index = np.argmax(class_box_scores)

                # add box, class, and score to output lists
                box_predictions.append(class_boxes[max_score_index])
                predicted_box_classes.append(current_class)
                predicted_box_scores.append(class_box_scores[max_score_index])

                # get iou scores for max box and each box in class_boxes
                ious = np.array([self._iou(
                    class_boxes[max_score_index], box) for box in class_boxes])

                # find all boxes with a iou greater than the threshold
                above_threshold = np.where(ious > self.nms_t)

                # remove boxes and their scores that fell above the threshold
                class_boxes = np.delete(
                    class_boxes, above_threshold, axis=0)
                class_box_scores = np.delete(
                    class_box_scores, above_threshold, axis=0)

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """
        folder_path: a string representing the path to the folder holding all
            the images to load
        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        # create output lists
        images = []
        image_paths = []

        # loop through images in directory
        for image in os.listdir(folder_path):
            # add image path to image_paths
            image_paths.append(folder_path + '/' + image)
            # add image to images
            images.append(cv2.imread(image_paths[-1]))

        return images, image_paths

    def preprocess_images(self, images):
        """
        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
              containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model Note: this can
                  vary by model
                input_w: the input width for the Darknet model Note: this can
                  vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
              original height and width of the images
                2 => (image_height, image_width)
        """
        dsize = (self.model.input.shape[1].value,
                 self.model.input.shape[2].value)
        pimages, image_shapes = [], []
        for image in images:
            pimages.append(cv2.resize(image,
                                      dsize=dsize,
                                      interpolation=cv2.INTER_CUBIC
                                      ) / 255)
            image_shapes.append(image.shape[0:2])

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored
        Displays the image with all boundary boxes, class names, and box
          scores (see example below)
            Boxes should be drawn as with a blue line of thickness 2
            Class names and box scores should be drawn above each box in red
                Box scores should be rounded to 2 decimal places
                Text should be written 5 pixels above the top left corner of
                  the box
                Text should be written in FONT_HERSHEY_SIMPLEX
                Font scale should be 0.5
                Line thickness should be 1
                You should use LINE_AA as the line type
            The window name should be the same as file_name
            If the s key is pressed:
                The image should be saved in the directory detections, located
                  in the current directory
                If detections does not exist, create it
                The saved image should have the file name file_name
                The image window should be closed
            If any key besides s is pressed, the image window should be closed
              without saving
        """
        for i, box in enumerate(boxes):

            # unpack box coordinates
            x1, y1, x2, y2 = box.astype(int)

            # create rectangle
            cv2.rectangle(img=image,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(255, 0, 0),
                          thickness=2)

            # add label
            cv2.putText(img=image,
                        text='{} {:.2f}'.format(
                            self.class_names[box_classes[i]], box_scores[i]),
                        org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        # show image
        cv2.imshow(file_name, image)

        # wait for key press
        key = cv2.waitKey(0)

        # if 's' key is pressed, add image to 'detections' directory
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        # close image
        cv2.destroyAllWindows()
