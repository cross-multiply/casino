import tempfile
import os
import logging
import io
import h5py

import numpy as np
import cv2

from PIL import Image
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K

log = logging.getLogger(__name__)


def h5py_binary_to_keras(input_data):
    K.clear_session()
    h5 = h5py.File(io.BytesIO(input_data), 'r')
    return load_model(h5)


class MaskedModel(object):
    MIN_PROB = 0.75

    def __init__(self, model, model_labels, masks_bot, masks_top):
        self.mapping = {
            0: 'bot',
            1: 'bot',
            2: 'bot',
            3: 'bot',
            4: 'top'
        }
        self.masks_bot = masks_bot
        self.masks_top = masks_top
        self.model = model
        # this should already be sorted but it doesn't hurt to make sure. Keras
        # sorts the names and you can zip it with the output
        self.model_labels = sorted(model_labels)

    @classmethod
    def generate_flattened_arr(cls, inp_img, mask):
        indexes = []
        start = None
        end = None
        mask_widths = []
        shape = inp_img.shape

        if len(shape) == 2:
            h, w = shape
        else:
            h, w, _ = shape

        for i in range(w):
            tmp = np.where(mask[:, i] != 0)[0]

            if len(tmp) == 0:
                if start != None:
                    end = i
                    break
                continue
            if start == None:
                start = i
            indexes.append(tmp)
            mask_widths.append(len(tmp))

        mask_height = int(np.mean(mask_widths))
        mask_height = min(mask_widths)

        if len(shape) == 2:
            out_arr = np.zeros((mask_height, end-start), np.uint8)
        else:
            out_arr = np.zeros((mask_height, end-start, 3), np.uint8)

        for i in range(start, end):
            idxs = indexes[i-start]

            out_arr[:, i-start] = inp_img[idxs.min():idxs.min()+mask_height, i]

        return out_arr

    def extract_strips(self, image_top, image_bot):
        strips = {}
        for i, n in self.mapping.items():
            if n == 'top':
                mask = self.masks_top[i]
                res = self.generate_flattened_arr(image_top, mask)
            else:
                mask = self.masks_bot[i]
                res = self.generate_flattened_arr(image_bot, mask)
            strips[i] = res
        return strips

    def process_cv(self, image_top, image_bot):
        strips = self.extract_strips(image_top, image_bot)

        test_datagen = ImageDataGenerator(rescale=1./255)

        # The tempfile approach is not great, but is self contained, and makes
        # the resizing and loading operations much easier vs an in memory only
        # approach
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, 'chips'))
            for i, img in strips.items():
                path = os.path.join(tmpdir, 'chips', '{}.png'.format(i))
                cv2.imwrite(path, img)

            test_gen = test_datagen.flow_from_directory(
                tmpdir,
                target_size=(14, 257),
                batch_size=1,
                class_mode=None,
                shuffle=False
            )

            test_count = test_gen.n

            test_gen.reset()
            pred = self.model.predict_generator(
                test_gen,
                steps=test_count,
                verbose=1
            )

        # this is the chip index, and a list of the labels and their probabilities
        raw_probabilities = {}

        for i, n in enumerate(pred):
            raw_probabilities[i] = list(zip(self.model_labels, n))

        # this is the chip index, the most likely model and it's probability as a tuple
        most_likely = {}
        for i, idx in enumerate(np.argmax(pred, axis=1)):
            most_likely[i] = (self.model_labels[idx], pred[i][idx])
        return most_likely, raw_probabilities

    def post_process_output(self, predictions):
        # this is hardcoded against 'no_chip' as not having a value, and 5
        # chips total

        num_chips = 5

        out = {}
        for i, n in predictions.items():
            if n[1] <= self.MIN_PROB:
                log.warn("Minimum Probability Error", i, n)

            if n[0] == 'no_chip':
                num_chips = i
                break
            out[i] = n

        return num_chips, out
