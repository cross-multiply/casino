import logging
import numpy as np
import cv2
import skimage.color
from .video_analysis import VideoFrameProcessor

log = logging.getLogger(__name__)

__all__ = ['frames_bgr_to_lab', 'frames_lab_to_bgr', 'BGSubtractPropogate', 'BGSubV1']


def frames_bgr_to_lab(frames):
    frames_lab = [cv2.cvtColor(x, cv2.COLOR_BGR2LAB) for x in frames]
    return frames_lab


def frames_lab_to_bgr(frames):
    frames_bgr = [cv2.cvtColor(x, cv2.COLOR_LAB2BGR) for x in frames]
    return frames_bgr


class BGSubtractPropogate(object):

    def __init__(self):
        self.bottom_thresh = 220
        self.KERNEL_SIZE = 5
        self.OPEN_ITERATIONS = 2
        self._pre_drop_frames = 3

    @classmethod
    def generate_post_drop_frames(cls, drop, slide_idx):
        fg_idx, bg_idx = VideoFrameProcessor.get_indexes(drop)
        post_frames = []
        for i in range(fg_idx[0], slide_idx[0]+1):
            arr = drop[i].jpeg_data
            if i == fg_idx[0]:
                start = fg_idx[1]
            else:
                start = 0
            if i == slide_idx[0]:
                end = slide_idx[1] + 1
            else:
                end = len(arr)
            for j in range(start, end):
                post_frames.append(VideoFrameProcessor.decode_jpeg(arr[j]))

        return post_frames

    def generate_pre_post_thresholds(self, drop, slide_idx):
        fg_idx, bg_idx = VideoFrameProcessor.get_indexes(drop)

        post_frames = self.generate_post_drop_frames(drop, slide_idx)
        post_ths = self.generate_thresholds(post_frames)

        # Not the prettiest, but it gets N frames prior to the current drop to
        # figure out the thresholds for a non-AOI background subtraction region
        pre_frames = []
        start = 0
        if fg_idx[1] < self._pre_drop_frames:
            if fg_idx[0] == 0:
                log.error("ERROR: Not enough frames to generate pre Non-AOI image")
            else:
                arr = drop[fg_idx[0]-1]
                start_offset = len(arr) - (self._pre_drop_frames - fg_idx[1])
                for n in arr[start_offset:]:
                    pre_frames.append(VideoFrameProcessor.decode_jpeg(n))
        else:
            start = fg_idx[1] - self._pre_drop_frames

        for i in range(start, fg_idx[1]+1):
            pre_frames.append(
                VideoFrameProcessor.decode_jpeg(drop[fg_idx[0]].jpeg_data[i])
            )

        pre_ths = self.generate_forward_threshold(pre_frames)

        return post_ths, pre_ths

    def generate_forward_threshold(self, frames):
        ths = []
        for i in range(1, len(frames)):
            diff = (cv2.absdiff(frames[i], frames[i-1]).sum(2) / 3).astype('uint8')
            monochrome = cv2.bilateralFilter(diff, 5, 17, 17)

            ret, th = cv2.threshold(monochrome, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ths.append(th)

        return ths

    def generate_thresholds(self, frames):
        ths = []
        for i in range(2, len(frames)+1):
            #i = 5
            #f1_lab = skimage.color.rgb2hsv(frames[-i])
            #f2_lab = skimage.color.rgb2hsv(frames[-i+1])
            #diff = (cv2.absdiff(f1_lab, f2_lab).sum(2) / 3).astype('uint8')
            diff = (cv2.absdiff(frames[-i], frames[-i+1]).sum(2) / 3).astype('uint8')
            monochrome = cv2.bilateralFilter(diff, 5, 17, 17)

            ret, th = cv2.threshold(monochrome, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ths.append(th)

        return ths

    def open_img(self, img):
        kernel = np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE), np.uint8)
        return cv2.dilate(img, kernel, iterations=self.OPEN_ITERATIONS)

    def frame0_filter(self, frame):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)
        # stats format: LEFT, TOP, WIDTH, HEIGHT, AREA
        non_aoi_labels = []
        aoi_labels = []
        h, w = frame.shape
        for i, n in enumerate(stats):
            if n[0] == 0 and n[1] == 0 and n[2] == w and n[3] == h:
                continue
            if n[1] > self.bottom_thresh:
                aoi_labels.append(i)
            else:
                non_aoi_labels.append(i)

        aoi_frame = np.zeros(frame.shape, dtype=np.uint8)
        for i in aoi_labels:
            aoi_frame[labels==i] = 255
        non_aoi_frame = np.zeros(frame.shape, dtype=np.uint8)
        for i in non_aoi_labels:
            non_aoi_frame[labels==i] = 255

        _, contours, _ = cv2.findContours(aoi_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cvx_hull = cv2.convexHull(np.concatenate(contours))
        aoi_img = cv2.drawContours(np.zeros(frame.shape, dtype=np.uint8), [cvx_hull], -1, (255), cv2.FILLED)

        _, contours, _ = cv2.findContours(non_aoi_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cvx_hull = cv2.convexHull(np.concatenate(contours))
        non_aoi_img = cv2.drawContours(np.zeros(frame.shape, dtype=np.uint8), contours, -1, (255), cv2.FILLED)
        #non_aoi_img = cv2.drawContours(np.zeros(frame.shape, dtype=np.uint8), [cvx_hull], -1, (255), cv2.FILLED)

        return aoi_img, non_aoi_img

    def frame_filter(self, frame, prev_aoi, prev_non_aoi):
        # Simple 'AND' removal of non AOI components.
        #tmp = np.zeros(frame.shape, dtype=np.uint8)
        #for n in prev_non_aoi:
        #    tmp |= n
        #frame_filtered = frame & cv2.bitwise_not(tmp)#cv2.bitwise_not(prev_non_aoi)
        frame_filtered = frame & cv2.bitwise_not(prev_non_aoi)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_filtered)
        # stats format: LEFT, TOP, WIDTH, HEIGHT, AREA
        non_aoi_labels = []
        aoi_labels = []
        h, w = frame.shape
        for i, n in enumerate(stats):
            if n[0] == 0 and n[1] == 0 and n[2] == w and n[3] == h:
                continue
            if (prev_aoi[labels == i].sum() > 0):
                aoi_labels.append(i)
            else:
                non_aoi_labels.append(i)

        aoi_frame = np.zeros(frame.shape, dtype=np.uint8)
        for i in aoi_labels:
            aoi_frame[labels==i] = 255
        non_aoi_frame = np.zeros(frame.shape, dtype=np.uint8)
        for i in non_aoi_labels:
            non_aoi_frame[labels==i] = 255

        _, contours, _ = cv2.findContours(aoi_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cvx_hull = cv2.convexHull(np.concatenate(contours))
        aoi_img = cv2.drawContours(np.zeros(frame.shape, dtype=np.uint8), [cvx_hull], -1, (255), cv2.FILLED)

        _, contours, _ = cv2.findContours(non_aoi_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cntrs = np.concatenate(contours)
        #epsilon = 0.1*cv2.arcLength(cntrs, True)
        #cvx_hull = cv2.approxPolyDP(cntrs, 0, True)
        # = cv2.convexHull()
        non_aoi_img = cv2.drawContours(np.zeros(frame.shape, dtype=np.uint8), contours, -1, (255), cv2.FILLED)

        return aoi_img, non_aoi_img


class BGSubV1(object):
    def __init__(self):
        self.BG_DELTA_SIZE = 15

    @classmethod
    def calibrate_frames(cls, frames_lab, thresholds, initial_offset):
        """This function assumes that frames are passed in FG -> BG"""

        global_l_adj = int(50 - initial_offset[0])
        global_a_adj = int(127 - initial_offset[1])
        global_b_adj = int(127 - initial_offset[2])

        log.info("Global {} {} {} {}".format(initial_offset, global_l_adj, global_a_adj, global_b_adj))
        ret = []
        w, h = thresholds[0].shape
        bg_frame = frames_lab[-1]

        for i in range(1, len(frames_lab)):
            bg = bg_frame
            fg = frames_lab[-i-1]
            thresh = thresholds[-i]

            target_pixels = thresh == 0

            (l_fg, a_fg, b_fg) = cv2.split(fg)
            (l_bg, _, _) = cv2.split(bg)

            a_fg = np.add(a_fg.astype('int16'), global_a_adj).clip(0, 255).astype('uint8')
            b_fg = np.add(b_fg.astype('int16'), global_b_adj).clip(0, 255).astype('uint8')
            l_fg = np.add(l_fg.astype('int16'), global_l_adj).clip(0, 255).astype('uint8')

            # TODO: If 0?
            if target_pixels.sum() > 0:
                fg_l_avg = l_fg[target_pixels].mean()
                bg_l_avg = l_bg[target_pixels].mean()

                diff = int(bg_l_avg - fg_l_avg)

                l_fg = l_fg.astype('int16')
                tmp = np.add(l_fg, diff)
                np.clip(tmp, 0, 255, out=tmp)
                l_fg = tmp.astype('uint8')
                #new_fg_avg = l_fg[target_pixels].mean()
            else:
                log.error("No non-changed pixels")

            calibrated_fg = cv2.merge([l_fg, a_fg, b_fg])
            #calibrated_fg_bgr = cv2.cvtColor(calibrated_fg, cv2.COLOR_LAB2BGR)

            bg_frame = calibrated_fg
            ret.append(calibrated_fg)

        # Finally, adjust the background frame
        (l_bg, a_bg, b_bg) = cv2.split(frames_lab[-1])
        a_bg = np.add(a_bg.astype('int16'), global_a_adj).clip(0, 255).astype('uint8')
        b_bg = np.add(b_bg.astype('int16'), global_b_adj).clip(0, 255).astype('uint8')
        l_bg = np.add(l_bg.astype('int16'), global_l_adj).clip(0, 255).astype('uint8')

        calibrated_bg = cv2.merge([l_bg, a_bg, b_bg])
        ret.insert(0, calibrated_bg)

        return ret

    def bg_subtract(self, bg, fg):
        fg_copy = fg.copy()

        bg_blur = cv2.GaussianBlur(bg, (0, 0), 2)
        fg_blur = cv2.GaussianBlur(fg, (0, 0), 2)

        #bg_lab, fg_lab = self.frames_bgr_to_lab([bg_blur, fg_blur])
        bg_lab = skimage.color.rgb2lab(bg_blur)
        fg_lab = skimage.color.rgb2lab(fg_blur)
        delta = np.linalg.norm(fg_lab - bg_lab, axis=2)

        fg_copy[delta < self.BG_DELTA_SIZE] = 0

        return fg_copy

    def bounding_box(self, subtracted):
        kernel = np.ones((20, 20), np.uint8)

        img_h, img_w, _ = subtracted.shape

        gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(smooth, 30, 255, cv2.THRESH_BINARY)


        # merge together disjoint components to avoid failing to detect contour of sufficient area
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # further eliminate noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # alternatively, we merge and then chop out noise
        close_then_open = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # sometimes contours attained from opening suffice, but when chips are
        # separated slightly we don't get a coherent stack
        _, contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, CO_contours, _ = cv2.findContours(close_then_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        merged = sorted(contours + CO_contours, key=lambda c: cv2.contourArea(c))

        x = y = w = h = None

        # we iterate in descending order of area to find tightest bounding box containing stack
        for c in merged:
            area = cv2.contourArea(c)
            area_percentage = area/(subtracted.shape[0] * subtracted.shape[1])
            # discard noise contours
            if area_percentage < 0.2:
                #print("Skipping since area is ", area_percentage)
                continue

            new_x, new_y, new_w, new_h = cv2.boundingRect(c)

            # skip bounding boxes of larger area
            if w and h and w * h < new_w * new_h:
                    continue

            x = new_x
            y = new_y
            w = new_w
            h = new_h
            #print(x,y,w,h)
            #cv2.rectangle(subtracted, (x,y), (x+w,y+h), (0, 255, 0), 2)

        # failed to find contour of sufficient area
        if not x and not y and not w and not h:

            # making heavy assumptions here
            log.error("Failed to crop via subtraction, using hardcoded left right", 100,540)
            return (0, img_h, 100, 540)

        # note this dimensions are quite weird, y+h is actually supposedly the base of the stack
        # sometimes we erode too much, so we need to make sure the base of the stack is captured
        base = 420
        # if y+h < 420:
        # 	print("FIXING BASE-----------------")
        # 	base = 420
        # else:
        # 	base = y+h


        # we can also make assumptions about the right of the stack
        right = None
        if x + w > 540:
            right = 540
        else:
            right = x + w

        return (y, base, x, right)
