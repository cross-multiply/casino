import logging
import json
import io
import zipfile
import zlib
import binascii
import os
import subprocess
import math
import cv2
import numpy as np
import tempfile

from chipcounting.image_processing.video_analysis import ImageGrouping, VideoFrameProcessor
from chipcounting.image_processing.bg_subtraction import BGSubtractPropogate, BGSubV1, \
    frames_bgr_to_lab, frames_lab_to_bgr

log = logging.getLogger(__name__)


class ChipUploadHelper(object):

    @classmethod
    def init_from_serialized(cls, data):
        try:
            data = zlib.decompress(data)
        except:
            log.debug("Data not in zlib format, trying direct json")

        tmp = json.loads(data)
        image_groupings = [ImageGrouping.deserialize(x) for x in tmp]
        return cls(image_groupings)

    def __init__(self, image_groupings):
        self.image_groupings = image_groupings

    def get_drop_images(self):
        p_idx, r_idx = VideoFrameProcessor.get_indexes(self.image_groupings)
        primed = self.image_groupings[p_idx[0]].jpeg_data[p_idx[1]]
        recovery = self.image_groupings[r_idx[0]].jpeg_data[r_idx[1]]

        return primed, recovery

    def generate_image_zipfile(self):
        i = 0

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_STORED, False) as zip_file:
            for group in self.image_groupings:
                for jpeg in group.jpeg_data:
                    zip_file.writestr('{:03d}.jpeg'.format(i), jpeg)
                    i += 1

        zip_buffer.seek(0)
        return zip_buffer.read()


class ChipUploadHelperV2(object):

    FRAME_BACKTRACK = 2/21
    @classmethod
    def _take_closest(cls, num, collection):
        return min(collection, key=lambda x: abs(x-num))

    @classmethod
    def extract_bottom(cls, inp):
        files = {}
        index = None
        with zipfile.ZipFile(io.BytesIO(inp), 'r') as bottom_zip:
            for file_name in bottom_zip.namelist():
                data = bottom_zip.read(file_name)
                if file_name == 'index.json':
                    index = json.loads(data)
                else:
                    files[file_name] = data
        # Sanity Checks
        assert index is not None
        assert index.keys() == files.keys(), "Bottom Camera Index and Files do not match"
        return index, files

    @classmethod
    def find_closest_times(cls, bottom_index, top_groupings):
        top_index_mapping = {}
        for idx, n in enumerate(top_groupings):
            start_time = n.start_time
            closest = cls._take_closest(start_time - cls.FRAME_BACKTRACK, list(sorted(bottom_index.values())))
            for i, v in bottom_index.items():
                if v == closest:
                    closest_frame = i
                    break
            top_index_mapping[idx] = (closest, closest_frame)
            log.debug("Grouping Idx: {} error of {:2f} seconds".format(idx, start_time - closest))

        return top_index_mapping

    @classmethod
    def regroup_frames(cls, top_groupings, bottom_index, bottom_frames):
        # There's some somewhat convoluted logic here to handle index and frames that are not placed in order.
        # That shouldn't happen, but it's better to build around it.
        closest_times = cls.find_closest_times(bottom_index, top_groupings)

        #bottom_times_sorted = sorted(bottom_index.values())
        bottom_sorted_with_frames = sorted(bottom_index.items(), key=lambda x: x[1])

        out = []
        for idx, n in enumerate(top_groupings):
            start_time = closest_times[idx][0]
            end_time = start_time + 1
            # We're going to assume that we are accurately capturing at 21 fps and not validate it.
            #take_closest(end_time, bottom_times_sorted) 
            #end_idx = bottom_times_sorted[start_idx + 21]

            # Get the frames that fit between the two times.
            frame_idxes = filter(lambda x: start_time <= x[1] <= end_time, bottom_sorted_with_frames)

            frames = []
            for key, frame_time in frame_idxes:
                frames.append((frame_time, bottom_frames[key]))
            out.append((n, frames))
        return out

    @classmethod
    def init_from_serialized(cls, data):
        try:
            data = zlib.decompress(data)
        except zlib.error:
            log.debug("Data not in zlib format, trying direct json")

        tmp = json.loads(data)

        top = [ImageGrouping.deserialize(x) for x in tmp['top']]
        bottom_zip = binascii.a2b_base64(tmp['bottom'])

        index, files = cls.extract_bottom(bottom_zip)
        frames = cls.regroup_frames(top, index, files)
        # Frames is an array consisting of the following structure
        # [(ImageGrouping, [(Frame Time, Bottom Frame), ....]), ...]

        bottom_images = [x[1] for x in frames]
        return cls(top, bottom_images)

    def __init__(self, top_image_groupings, bottom_image_groupings):
        # Bottom frames may not cover all the top images
        self.top_image_groupings = top_image_groupings
        self.bottom_image_groupings = bottom_image_groupings

    def get_drop_images(self):
        p_idx, r_idx = VideoFrameProcessor.get_indexes(self.top_image_groupings)

        primed = self._gen_combined_cv_image(p_idx[0], p_idx[1])
        recovery = self._gen_combined_cv_image(r_idx[0], r_idx[1])

        # This returns a status code and a numpy array, which we can cast
        # directly to a bytes object
        _, primed = cv2.imencode('.jpg', primed)
        _, recovery = cv2.imencode('.jpg', recovery)
        return bytes(primed), bytes(recovery)

    def get_cv_images(self):
        p_idx, r_idx = VideoFrameProcessor.get_indexes(self.top_image_groupings)

        bot_idx = self._top_idx_to_bot_idx(p_idx[1])
        primed_top = self.top_image_groupings[p_idx[0]].jpeg_data[p_idx[1]]
        primed_bot = self.bottom_image_groupings[p_idx[0]][bot_idx][1]

        primed_top = VideoFrameProcessor.decode_jpeg(primed_top)
        primed_bot = VideoFrameProcessor.decode_jpeg(primed_bot)
        bot_idx = self._top_idx_to_bot_idx(r_idx[1])
        recovery_top = self.top_image_groupings[r_idx[0]].jpeg_data[r_idx[1]]
        recovery_bot = self.bottom_image_groupings[r_idx[0]][bot_idx][1]

        recovery_top = VideoFrameProcessor.decode_jpeg(recovery_top)
        recovery_bot = VideoFrameProcessor.decode_jpeg(recovery_bot)
        return (primed_top, primed_bot), (recovery_top, recovery_bot)

    def _top_idx_to_bot_idx(self, i, bot_top_factor=2):
        return math.floor(i / bot_top_factor)

    def _gen_combined_cv_image(self, offset, idx, h=480, w=640):
        bot_idx = self._top_idx_to_bot_idx(idx)
        out_dim = (h, w*2, 3)

        top_frame = VideoFrameProcessor.decode_jpeg(self.top_image_groupings[offset].jpeg_data[idx])

        bot_group = self.bottom_image_groupings[offset]
        out_frame = np.zeros(out_dim, np.uint8)
        out_frame[:, :640, :] = top_frame
        if bot_idx >= len(bot_group):
            log.errors("Not enough Bottom Frames {}/{}! inserting blank".format(bot_idx, len(bot_group)))
        else:
            bot_frame = VideoFrameProcessor.decode_jpeg(bot_group[bot_idx][1])
            out_frame[:, 640:, :] = bot_frame

        return out_frame

    def generate_combined_jpegs(self, h=480, w=640):
        # TODO Change function to use function above
        out_frames = []

        out_dim = (h, w*2, 3)

        for n in zip(self.top_image_groupings, self.bottom_image_groupings):
            i = 0
            for i, frame in enumerate(n[0].jpeg_data):
                top_frame = VideoFrameProcessor.decode_jpeg(frame)
                bot_idx = self._top_idx_to_bot_idx(i)

                out_frame = np.zeros(out_dim, np.uint8)
                out_frame[:, :640, :] = top_frame
                if bot_idx >= len(n[1]):
                    print("Not enough Bottom Frames {}/{}! inserting blank".format(bot_idx, len(n[1])))
                else:
                    bot_frame = VideoFrameProcessor.decode_jpeg(n[1][bot_idx][1])
                    out_frame[:, 640:, :] = bot_frame

                out_frames.append(out_frame)
        return out_frames

    def generate_drop_video(self):
        fps = 42
        fd, fname = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(fname, fourcc, fps, (480, 640 * 2))
        for n in self.generate_combined_jpegs():
            video.write(n)
        video.release()

        data = open(fname, 'rb').read()
        os.unlink(fname)
        return data

    def generate_image_zipfile(self):
        i = 0

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_STORED, False) as zip_file:
            for frame in self.generate_combined_jpegs():
                zip_file.writestr('{:03d}.jpeg'.format(i), frame)
                i += 1

        zip_buffer.seek(0)
        return zip_buffer.read()


class DebugChipDataHelper(ChipUploadHelper):

    @classmethod
    def init_from_serialized(cls, data):
        try:
            data = zlib.decompress(data)
        except zlib.error:
            log.debug("Data not in zlib format, trying direct json")

        tmp = json.loads(data)
        assert tmp['version'] == 1
        image_json = tmp['image_groupings']
        slide_idx = tmp['slide_idx']
        image_groupings = [ImageGrouping.deserialize(x) for x in image_json]

        try:
            fg_img = binascii.a2b_base64(tmp['calibrated_frames']['fg'])
            slide_img = binascii.a2b_base64(tmp['calibrated_frames']['slide'])
        except KeyError:
            fg_img = None
            slide_img = None

        c = cls(image_groupings, slide_idx)
        c._calibrated_fg_img = fg_img
        c._calibrated_slide_img = slide_img

        return c

    def __init__(self, image_groupings, slide_idx):
        self.image_groupings = image_groupings
        self.slide_idx = slide_idx
        self._bgp = BGSubtractPropogate()
        self._bgsub_v1 = BGSubV1()
        self._calibrated_frames = None
        self._calibrated_fg_img = None
        self._calibrated_slide_img = None

    def color_calibrate_drop(self):
        """ Returns calibrated frames in the following order"""
        si_group, si_frame = self.slide_idx
        found_frame, calibration_data = self.image_groupings[si_group].calibration_data[si_frame]

        post_ths, pre_ths = self._bgp.generate_pre_post_thresholds(self.image_groupings, self.slide_idx)
        frames = self._bgp.generate_post_drop_frames(self.image_groupings, self.slide_idx)

        frames_lab = frames_bgr_to_lab(frames)
        calibrated_lab = self._bgsub_v1.calibrate_frames(frames_lab, post_ths, calibration_data)
        calibrated = frames_lab_to_bgr(calibrated_lab)

        self._calibrated_frames = list(reversed(calibrated))
        self._calibrated_slide_img = VideoFrameProcessor.opencv_to_jpeg(calibrated[0])
        self._calibrated_fg_img = VideoFrameProcessor.opencv_to_jpeg(calibrated[-1])

        return calibrated

    def generate_data_field(self, compress=True):
        out = {
            'version': 1,
            'image_groupings': [ImageGrouping.serialize(x) for x in self.image_groupings],
            'slide_idx': self.slide_idx,
        }

        if self._calibrated_slide_img is not None:
            out['calibrated_frames'] = {
                'fg': binascii.b2a_base64(self._calibrated_fg_img, newline=False).decode('ascii'),
                'slide': binascii.b2a_base64(self._calibrated_slide_img, newline=False).decode('ascii'),
            }

        out_data = json.dumps(out)
        if compress:
            out_data = zlib.compress(out_data.encode('ascii'))

        return out_data
