import logging
import time
import cv2
import numpy as np
import io
import zlib
import ast
import binascii

from .config import get_local_config

log = logging.getLogger(__name__)


class ParamHelper():

    @classmethod
    def load_int_or_default(cls, params, key, default):
        try:
            return int(params.get(key, default))
        except Exception:
            return default

    @classmethod
    def load_float_or_default(cls, params, key, default):
        try:
            return float(params.get(key, default))
        except Exception:
            return default

    @classmethod
    def load_ast_or_default(cls, params, key, default):
        try:
            return ast.literal_eval(params.get(key, default))
        except Exception:
            return default


class CalibrationDetector():

    candidates = None
    outer_box = None
    outer_mask_cut = None
    inner_mask_cut = None
    outer_mask = None
    inner_mask = None
    upper_bound = 280
    lower_bound = 400
    debug = False

    # Parameters
    # XXX: This needs to be a lower number for the updated camera position, we filter the excess contours in a later phase.
    ADAPTIVE_THRESHOLD_KERNEL_SIZE = 19 #29
    ADAPTIVE_THRESHOLD_PARAM_C = 1
    CALIBRATION_AREA_RATIO = 0.7  # innerArea*16 = outerArea*23
    CALIBRATION_AREA_RATIO_THRESHOLD = 0.25
    MASK_ERODE_KERNEL_SIZE = 3
    MASK_ERODE_KERNEL_ITERATIONS = 2
    CALIBRATION_MIN_LIGHT = 15  # L in LAB
    CALIBRATION_RATIO = 1.1  # outer_L > 1.3 * inner_L is ideal, we'll use 10% since we do other checks as well.
    CALIBRATION_STD_DEV_MAX = 6.375*1.10  # setting to 2.5% => 2.5*255/100. = 6.375
    CALIBRATION_STD_DEV_LIGHT_MAX = 12.75*1.5
    CONTOUR_MIN_AREA = 1000  # minimum inner area for filtering noise
    SLIDE_DETECTION_GRAY_RATIO = 1.10
    SLIDE_DETECTION_MAX_LIGHT = 35

    # Area slice to check if the slide is open or not
    # Y = 392:425, X = 125:530
    SLIDE_BBOX = ((392, 425), (125, 530))

    def _load_params_from_thread_local(self):
        thread_local = get_local_config()
        params = None

        if hasattr(thread_local, 'cv_params') and 'calibration_detector' in thread_local.cv_params:
            params = thread_local.cv_params['calibration_detector']
            log.debug("Using thread local parameters: {}".format(params))
            try:
                self.CALIBRATION_MIN_LIGHT = ParamHelper.load_int_or_default(params, 'min_light', 15)
                self.CALIBRATION_RATIO = ParamHelper.load_float_or_default(params, 'calibration_ratio', 1.1)
                self.SLIDE_DETECTION_GRAY_RATIO = ParamHelper.load_float_or_default(params, 'slide_ratio', 1.1)
                self.SLIDE_MAX_LIGHT = ParamHelper.load_int_or_default(params, 'slide_max_light', 35)
                self.SLIDE_BBOX = ParamHelper.load_ast_or_default(params, 'slide_bbox', ((392, 425), (125, 520)))
                self.lower_bound = ParamHelper.load_int_or_default(params, 'lower_bound', 400)
                self.upper_bound = ParamHelper.load_int_or_default(params, 'upper_bound', 280)
            except Exception as e:
                log.exception(e)

    def __init__(self, debug=False):
        self.candidates = None
        self.outer_box = None
        self.outer_mask_cut = None
        self.outer_outer_mask_cut = None
        self.inner_mask_cut = None
        self.outer_mask = None
        self.outer_outer_mask = None
        self.inner_mask = None
        self.debug = True # TODO: Change back to the input param
        self._load_params_from_thread_local()

    def serialize_state(self):
        if not self.is_locked:
            return None

        return {
            'omc': self.outer_mask_cut.copy(),
            'oomc': self.outer_outer_mask_cut.copy(),
            'imc': self.inner_mask_cut.copy(),
            'om': self.outer_mask.copy(),
            'oom': self.outer_outer_mask.copy(),
            'im': self.inner_mask.copy(),
            'ob': self.outer_box
        }

    @classmethod
    def decompress_serialized(cls, inp):
        if inp is None:
            return None

        ret = {}
        for key in inp:
            data = zlib.decompress(binascii.a2b_base64(inp[key]))
            tt = io.BytesIO(data)
            loaded = np.load(tt, allow_pickle=False)
            ret[key] = loaded

        return ret

    @classmethod
    def compress_serialized(cls, inp):
        out_ar = {}

        for key in inp.keys():
            tt = io.BytesIO()
            np.save(tt, inp[key], allow_pickle=False)
            tt.seek(0)
            data = tt.read()
            data = binascii.b2a_base64(zlib.compress(data), newline=False).decode('ascii')
            out_ar[key] = data

        return out_ar

    def load_state(self, state):
        if state is None:
            return
        self.inner_mask_cut = state['imc']
        self.outer_mask_cut = state['omc']
        self.outer_outer_mask_cut = state['oomc']
        self.outer_mask = state['om']
        self.outer_outer_mask = state['oom']
        self.inner_mask = state['im']
        self.outer_box = state['ob']

    @property
    def is_locked(self):
        return self.outer_box is not None

    def filter_contour_hierarchy_candidates(self, hierarchy):
        """ Hierarchy Format:
        [next, previous, first_child, parent]

        """
        innermost = list(filter(lambda x: x[1][2] == -1 and x[1][3] != -1, enumerate(hierarchy)))
        candidates = []
        seen_combos = set()
        for i, n in innermost:
            # This is the bounding box of the gray region
            parent1 = n[3]
            # Check is already taken care of by the filter
            # TODO: This was commented out... Double check why.
            if parent1 == -1:
                continue

            # This is the bounding box of the white region
            parent2 = hierarchy[parent1][3]
            if parent2 == -1:
                continue

            #parent3 = hierarchy[parent2][3]
            #if parent3 == -1:
            #    continue

            # It is possible to see multiple of the same parent combo due to
            # minor thresholding bright spots.
            if (parent1, parent2) in seen_combos:
                continue
            seen_combos.add((parent1, parent2))

            chain = [i, parent1, parent2]
            candidates.append((i, parent1, parent2))

            next_ = hierarchy[parent2][3]
            while next_ != -1:
                chain.append(next_)
                next_ = hierarchy[next_][3]
                candidates.append((chain[-3], chain[-2], chain[-1]))
            # i = inner box
            # parent1 = gray region box
            # parent2 = white region box

        return candidates

    def check_contour_ratios(self, contours, inner_idx, outer_idx):
        thresh = self.CALIBRATION_AREA_RATIO_THRESHOLD
        ratio = self.CALIBRATION_AREA_RATIO

        # 270, 121 => 0.44
        # 240, 90 => 0.375
        outer_box_wh_ratio = 0.44
        inner_box_wh_ratio = 0.375
        inner_area = cv2.contourArea(contours[inner_idx])
        if inner_area < self.CONTOUR_MIN_AREA:
            return False

        outer_area = cv2.contourArea(contours[outer_idx])

        # This shouldn't be needed, but sometimes the outermost can be a 0 size one.
        # I think this is a bug in OpenCV
        if outer_area < self.CONTOUR_MIN_AREA:
            return False

        if np.allclose(inner_area, outer_area*ratio, rtol=thresh, atol=0):
            i_com, i_wh, i_angle = cv2.minAreaRect(contours[inner_idx])
            o_com, o_wh, o_angle = cv2.minAreaRect(contours[outer_idx])

            com = np.allclose(
                (i_com[0], i_com[1]),
                (o_com[0], o_com[1]),
                rtol=0.1,
                atol=0
            )
            # return True if com else False
            # Angles can very quite a bit (positive for one negative for the other),
            # commenting out until later testing
            # angle = np.allclose(i_angle, o_angle, rtol=0.2, atol=3)

            tmp = i_wh[0]/i_wh[1]
            tmp2 = i_wh[1]/i_wh[0]

            i_wh_lower = tmp if tmp < tmp2 else tmp2
            tmp = o_wh[0]/o_wh[1]
            tmp2 = o_wh[1]/o_wh[0]

            o_wh_lower = tmp if tmp < tmp2 else tmp2

            i_ratio_close = 0.30 < i_wh_lower < 0.55
            o_ratio_close = 0.30 < o_wh_lower < 0.55

            close = com and i_ratio_close and o_ratio_close

            return True if close else False

            #wh_ratio_inner = np.isclose(
            #    (i_wh[0]/i_wh[1], i_wh[0]/i_wh[1]),
            #    (inner_box_wh_ratio, inner_box_wh_ratio),
            #    rtol=0.2, atol=0.1
            #)

            #wh_ratio_outer = np.isclose(
            #    (o_wh[0]/o_wh[1], o_wh[0]/o_wh[1]),
            #    (outer_box_wh_ratio, outer_box_wh_ratio),
            #    rtol=0.2, atol=0.1
            #)

            # wh_ratio_neg = np.allclose(
            #     (o_wh[0]/o_wh[1], i_wh[0]/i_wh[1]),
            #     (outer_box_wh_ratio, inner_box_wh_ratio),
            #     rtol=0.2, atol=0.1
            # )
            # wh_ratio_pos = np.allclose(
            #     (o_wh[1]/o_wh[0], i_wh[1]/i_wh[0]),
            #     (outer_box_wh_ratio, inner_box_wh_ratio),
            #     rtol=0.2, atol=0.1
            # )

            # ratio_close = wh_ratio_neg or wh_ratio_pos
            # close = com and angle and ratio_close

            # if not close:
            #     log.debug("Filtering based on COM/Angle and/or W/H Ratios: {}:{} {} {} {}".format(
            #         com, angle, ratio_close, (i_com, i_wh, i_angle), (o_com, o_wh, o_angle)
            #     ))
            # return True if close else False
        return False

    def filter_and_get_contours(self, inp_image):
        cut_img = inp_image[self.upper_bound:self.lower_bound, :, :]
        gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)

        # Kernel size was 11 before, but it increases the complexity
        monochrome = cv2.bilateralFilter(gray, 5, 17, 17)
        th = cv2.adaptiveThreshold(
            monochrome, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.ADAPTIVE_THRESHOLD_KERNEL_SIZE, self.ADAPTIVE_THRESHOLD_PARAM_C
        )
        # ret, th = cv2.threshold(monochrome, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # canny = cv2.Canny(monochrome, 30, 200)
        contours, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hier[0]

    def find_markers(self, inp_image):
        contours, hier = self.filter_and_get_contours(inp_image)
        candidates = self.filter_contour_hierarchy_candidates(hier)
        valid_candidates = [
            (contours[x[1]], contours[x[2]]) for x in candidates if self.check_contour_ratios(contours, x[1], x[2])
        ]
        #valid_candidates = list(filter(lambda x: self.check_contour_ratios(contours, x[1], x[2]), candidates))
        return valid_candidates

    def generate_masks(self, inp_img, inner_cnt, outer_cnt):
        ub = self.upper_bound
        lb = self.lower_bound
        mask1 = np.zeros((inp_img.shape[0], inp_img.shape[1]), dtype='uint8')
        mask_cut = np.zeros((lb-ub, inp_img.shape[1]), dtype='uint8')

        outer_outer_mask_tmp = mask1.copy()
        outer_outer_mask_tmp[ub:lb, :] = cv2.drawContours(mask_cut.copy(), [outer_cnt], -1, 255, cv2.FILLED)
        outer_outer_mask = mask1.copy()

        outer_mask = mask1.copy()
        outer_mask[ub:lb, :] = cv2.drawContours(mask_cut.copy(), [outer_cnt, inner_cnt], -1, 255, cv2.FILLED)
        inner_mask = mask1.copy()
        inner_mask[ub:lb, :] = cv2.drawContours(mask_cut, [inner_cnt], -1, 255, cv2.FILLED)

        kernel = np.ones((self.MASK_ERODE_KERNEL_SIZE, self.MASK_ERODE_KERNEL_SIZE), np.uint8)
        # The outer mask is so thin we frequently will lose it completely if we erode it
        #outer_mask[ub:lb, :] = cv2.erode(outer_mask[ub:lb, :], kernel, iterations=self.MASK_ERODE_KERNEL_ITERATIONS)
        inner_mask[ub:lb, :] = cv2.erode(inner_mask[ub:lb, :], kernel, iterations=self.MASK_ERODE_KERNEL_ITERATIONS)
        # Generate a mask of the surrounding black by taking the outline of the
        # outer mask, increasing it's size, Boring with itself and eroding it
        outer_outer_mask[ub:lb, :] = cv2.dilate(outer_outer_mask_tmp[ub:lb, :], kernel, iterations=5)
        outer_outer_mask ^= outer_outer_mask_tmp
        outer_outer_mask[ub:lb, :] = cv2.erode(outer_outer_mask[ub:lb, :], kernel, iterations=1)

        return inner_mask, outer_mask, outer_outer_mask

    def get_calibration_parameters(self, img, inner, outer, outer_outer):
        clr = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        mean_in, stddev_in = cv2.meanStdDev(clr, mask=inner)
        mean_out, stddev_out = cv2.meanStdDev(clr, mask=outer)
        mean_out_out, stddev_out_out = cv2.meanStdDev(clr, mask=outer_outer)

        inner_L = mean_in[0][0]*100/255.
        outer_L = mean_out[0][0]*100/255.
        outer_outer_L = mean_out_out[0][0]*100/255.
        is_light_enough = outer_L > self.CALIBRATION_MIN_LIGHT
        is_lighter = outer_L > inner_L * self.CALIBRATION_RATIO
        is_darker = inner_L > outer_outer_L * self.CALIBRATION_RATIO  # Set a low standard of just darker
        STD_DEV_MAX = self.CALIBRATION_STD_DEV_MAX
        LIGHT_STD_DEV_MAX = self.CALIBRATION_STD_DEV_LIGHT_MAX
        std_dev_low = np.all(stddev_in[1:] < STD_DEV_MAX) and \
                      np.all(stddev_out[1:] < STD_DEV_MAX) and \
                      stddev_in[0][0] < LIGHT_STD_DEV_MAX and  \
                      stddev_out[0][0] < LIGHT_STD_DEV_MAX #and  \
                      # Removing the outer check since the LED reflection varies too much
                      # and can affect the quality
                      #np.all(stddev_out_out[1:] < STD_DEV_MAX) and \
                      #stddev_out_out[0][0] < LIGHT_STD_DEV_MAX

        is_similar = all([is_lighter, is_darker, std_dev_low, is_light_enough])
        if not is_similar and self.debug:
            log.debug(
                "Not similar %s %s %s",
                [is_lighter, is_darker, std_dev_low, is_light_enough],
                [inner_L, outer_L, outer_outer_L],
                [stddev_in, stddev_out]
            )
        return is_similar, mean_in

    def color_shift(self, img, calib_lab):
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l = np.uint8(128 - calib_lab[0][0])
        a = np.uint8(128 - calib_lab[1][0])
        b = np.uint8(128 - calib_lab[2][0])
        converted[:, :, 0] += l
        converted[:, :, 1] += a
        converted[:, :, 2] += b
        return cv2.cvtColor(converted, cv2.COLOR_LAB2BGR)

    def get_slide_lab(self, frame):
        bbox = self.SLIDE_BBOX
        cut = frame[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        converted = cv2.cvtColor(cut, cv2.COLOR_BGR2LAB)
        mean, stddev = cv2.meanStdDev(converted)
        return mean, stddev

    def check_for_slide(self, frame, calib_params):
        slide_mean, slide_stddev = self.get_slide_lab(frame)
        calib_L = calib_params[0][0]
        slide_L = slide_mean[0][0]

        # Aim for 5% brighter
        # Calib_L*100/255 > (Slide_L*105/255) * 1.05
        is_lighter = calib_L > slide_L * self.SLIDE_DETECTION_GRAY_RATIO
        is_dark_enough = slide_L*100/255. < self.SLIDE_DETECTION_MAX_LIGHT
        std_dev_low = np.all(slide_stddev < self.CALIBRATION_STD_DEV_LIGHT_MAX)

        return is_lighter and std_dev_low and is_dark_enough

    def quick_check_frame(self, frame):
        x, y, w, h = self.outer_box
        frame_cut = frame[y:y+h, x:x+w]
        is_similar, calib_lab = self.get_calibration_parameters(
            frame_cut, self.inner_mask_cut, self.outer_mask_cut,
            self.outer_outer_mask_cut
        )

        return is_similar, calib_lab

    def process_frame(self, frame):
        s = time.time()

        candidates = self.find_markers(frame)
        if len(candidates) > 1:
            log.debug("Too many candidates, taking the first: {} Total".format(len(candidates)))
            candidates = [candidates[0]]
        if len(candidates) == 0:
            if self.debug:
                log.debug("Found no candidates!")

            if self.outer_box is None:
                return None, None
        else:
            inner_mask, outer_mask, outer_outer_mask = self.generate_masks(
                frame, candidates[0][0], candidates[0][1]
            )
            self.inner_mask = inner_mask
            self.outer_mask = outer_mask
            self.outer_outer_mask = outer_outer_mask
            x, y, w, h = cv2.boundingRect(candidates[0][1])
            # Expand by 20 pixels allow for outer outer mask
            x -= 10
            y -= 10
            w += 10
            h += 10
            y += self.upper_bound
            self.outer_box = (x, y, w, h)
            self.outer_outer_mask_cut = outer_outer_mask[y:y+h, x:x+w]
            self.outer_mask_cut = outer_mask[y:y+h, x:x+w]
            self.inner_mask_cut = inner_mask[y:y+h, x:x+w]

        is_similar, calib_lab = self.quick_check_frame(frame)
        if self.debug:
            log.debug("Time: {0:.5f}".format(time.time() - s))

        return is_similar, calib_lab

    def annotate_image(self, frame):
        # should only use if "is similar"
        # will end up throwing exceptions if not / no calibration sticker was ever found
        x,y,w,h = self.outer_box
        annotated = cv2.circle(frame.copy(), (int(x+w/2), int(y+h/2)), 5, (0, 255, 0), 10)

        return annotated

