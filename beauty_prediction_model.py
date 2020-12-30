import logging

import cv2
import numpy as np
import tensorflow as tf
from deepface.commons.functions import detect_face

logging.getLogger().setLevel(logging.INFO)


class BeautyPredictionModel:

    basewidth = 800
    basewithfinal = 256

    def __init__(self, model_path):
        self.m = tf.keras.models.load_model(model_path)

    def extract_face(self, array):

        logging.log("try to extract face from image with shape " + str(array.shape))

        try:

            # scale image down otherwise face detection algorithm would be very slow
            wpercent = self.basewidth / float(array.shape[1])
            hsize = int((float(array.shape[0]) * float(wpercent)))
            img = cv2.resize(array, dsize=(self.basewidth, hsize))

            img_face = detect_face(
                img, detector_backend="mtcnn", enforce_detection=True
            )

            # create black square around image
            h, w, _ = img_face.shape
            m = max(w, h)
            square = np.zeros((m, m, 3), np.float32)
            h_start = int((m - h) / 2)
            w_start = int((m - w) / 2)
            h_end = h_start + h
            w_end = w_start + w
            square[h_start:h_end, w_start:w_end] = img_face

            # scale down
            res_image = cv2.resize(
                square, dsize=(self.basewithfinal, self.basewithfinal)
            )

            res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
            logging.info(f"face extracted")
            return res_image

        except Exception as e:
            logging.warning("error extracting face: " + str(e))

        return self.res_image

    def make_prediction(self, face_numpy):

        if face_numpy.shape[0] != self.basewidth:
            face_numpy = self.extract_face(face_numpy)

        try:
            if len(face_numpy.shape) == 3:
                face_numpy = face_numpy[np.newaxis, ...]
            score = self.m.predict(face_numpy).flatten()[0]
            logging.info("score " + str(score))
            return score
        except Exception as e:
            logging.warning("error making prediction: " + str(e))
            return -1
