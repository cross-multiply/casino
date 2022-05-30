import logging
import time
import threading
import queue
import cv2
import numpy as np
import binascii
import io

from PIL import Image
from collections import deque

from .calibration_detector import CalibrationDetector, ParamHelper
from .config import get_local_config

__all__ = [
    'ImageGrouping',
    'ProcessingQueue',
    'VideoFrameProcessor'
]

log = logging.getLogger(__name__)


class VideoFrameProcessor():
    SAD_MIN = 500
    MOTION_MIN = 10
    MIN_TOTAL_LENGTH = 6  # 140ms at 42fps
    MIN_MOTION_LENGTH = 1
    NUM_CONSECUTIVE_FRAMES = 1
    CUT_POINT = 336

    def _load_params_from_thread_local(self):
        thread_local = get_local_config()
        params = None

        if hasattr(thread_local, 'cv_params') and 'video_frame_processor' in thread_local.cv_params:
            params = thread_local.cv_params['video_frame_processor']
            try:
                self.MOTION_MIN = ParamHelper.load_int_or_default(params, 'motion_min', 10)
                self.MIN_TOTAL_LENGTH = ParamHelper.load_int_or_default(params, 'cap_min_total_length', 6)
                self.MIN_MOTION_LENGTH = ParamHelper.load_int_or_default(params, 'motion_min_length', 1)
                self.CUT_POINT = ParamHelper.load_int_or_default(params, 'motion_cut_point', 336)
            except Exception as e:
                log.exception(e)

    def __init__(self):
        self._load_params_from_thread_local()

    @classmethod
    def get_indexes(self, slide_pull):
        primed_idx = None
        recovery_idx = None
        for i, n in enumerate(slide_pull):
            if n.primed_idx is not None:
                primed_idx = (i, n.primed_idx)
            if n.recovery_idx is not None:
                recovery_idx = (i, n.recovery_idx)
        return primed_idx, recovery_idx

    @classmethod
    def opencv_to_jpeg(cls, inp):
        rgb_data = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()

        Image.fromarray(rgb_data).save(buf, format='JPEG')
        buf.seek(0)
        return buf.read()

    @classmethod
    def decode_jpeg(cls, inp):
        return cv2.imdecode(np.frombuffer(inp, dtype=np.uint8), cv2.IMREAD_COLOR)

    def generate_new_calibrations(self, caps, state):
        cd1 = CalibrationDetector()
        cd2 = CalibrationDetector()
        cd1.load_state(state)
        quick_arr = list()
        slow_arr = list()
        motion_data = list()

        last_frame = None

        s = time.time()
        frames = 0
        for image_grouping in caps:
            quick = list()
            slow = list()
            motion = list()
            for jpeg_bytes in image_grouping.jpeg_data:
                frames += 1
                jpeg = self.decode_jpeg(jpeg_bytes)
                if last_frame is None:
                    last_frame = jpeg

                motion.append(self._diff_frame_for_motion(jpeg, last_frame))
                last_frame = jpeg

                similar_quick, calib_quick = cd1.quick_check_frame(jpeg)
                similar_slow, calib_slow = cd2.process_frame(jpeg)
                drop_quick = False
                drop_slow = False
                if similar_quick:
                    drop_quick = cd1.check_for_slide(jpeg, calib_quick)
                if similar_slow:
                    drop_slow = cd1.check_for_slide(jpeg, calib_slow)

                quick.append((similar_quick, drop_quick))
                slow.append((similar_slow, drop_slow))
            motion_data.append(motion)
            quick_arr.append(quick)
            slow_arr.append(slow)
        log.info("Reprocessed {} Frames in {:.2f} seconds".format(frames, time.time() - s))
        return quick_arr, slow_arr, motion_data

    def _diff_frame_for_motion(self, inp1, inp2):
        cut_point = self.CUT_POINT
        arr1 = inp1[cut_point:, :]
        arr2 = inp2[cut_point:, :]

        val = cv2.norm(arr1, arr2, cv2.NORM_L1)
        val = val / arr1.size

        return val

    @classmethod
    def generate_motion_parameters(cls, first_frame, frames, cut_point=336):
        diffs = []
        prev_frame = first_frame[cut_point:]
        size = prev_frame.size
        for frame in frames:
            frame_cut = frame[cut_point:]
            val = cv2.norm(frame_cut, prev_frame, cv2.NORM_L1)
            val = val / size
            diffs.append(val)
            prev_frame = frame_cut

        return diffs

    @classmethod
    def process_motion_array(cls, arr):
        motion = (np.sqrt(
            np.square(arr['x'].astype(np.float)) +
            np.square(arr['y'].astype(np.float))
        ).clip(0, 255).astype(np.uint8) > cls.MOTION_MIN)*1
        sad = arr['sad'] > cls.SAD_MIN

        # grab lower frame
        # 20 => 16*20+16 = 336 pixels on
        # 22 => 16*22+16 = 368 pixels on
        motion_cut = (motion + sad)[:, 20:, :]

        # and summarize by array
        motion_sum = motion_cut.sum(1).sum(1)
        return motion_sum

    @classmethod
    def search_for_events(cls, primed, calibration_found_arr, motion_found_arr, primed_cntr=0, recovery_cntr=0):
        primed_idx = None
        recovery_idx = None
        for i, (has_calib, has_motion) in enumerate(zip(calibration_found_arr, motion_found_arr)):
            p_trig = False
            r_trig = False
            # Does not Have Motion. Has Calibration. Not currently in an active
            # state OR: Currently has motion and is not currently active
            if (not has_motion and has_calib and not primed) or (has_motion and not primed):
                continue
            # Does not have motion, Cannot Find Calibration Index, And We did
            # not find our "end".
            # The end should be taken care of by the break below
            elif not has_motion and not has_calib and not recovery_idx:
                p_trig = True
                if primed_cntr == cls.NUM_CONSECUTIVE_FRAMES:
                    primed = True
                    primed_idx = i
                else:
                    primed_cntr += 1
            # If we do not have motion, we have the calibration mark, we are in
            # an active state, and we did not find our "end"
            elif not has_motion and has_calib and primed and not recovery_idx:
                r_trig = True
                if recovery_cntr == cls.NUM_CONSECUTIVE_FRAMES:
                    recovery_idx = i
                    primed = False
                    break
                else:
                    recovery_cntr += 1
            # In an active state and motion detected. AKA do nothing.
            elif has_motion and primed:
                continue

            if not p_trig:
                primed_cntr = 0
            if not r_trig or recovery_idx is not None:
                recovery_cntr = 0
        return primed, primed_idx, recovery_idx, primed_cntr, recovery_cntr

    @classmethod
    def verify_has_motion(cls, groupings):
        p_idx, bg_idx = cls.get_indexes(groupings)
        motion_count = 0
        length = (bg_idx[0] - p_idx[0])*42 + bg_idx[1] - p_idx[1]
        if length < cls.MIN_TOTAL_LENGTH:
            log.info("Too few frames: {}".format(length))
            return False

        for grouping_idx in range(p_idx[0], bg_idx[0]+1):
            motion = groupings[grouping_idx].motion_data
            # start = p_idx[1] if grouping_idx == p_idx[0] else 0
            # end = bg_idx[1]+1 if grouping_idx == bg_idx[1] else len(motion)
            # motion_count += (np.array(motion[start:end]) > cls.).sum()

            # This should be using the code above, but to be safe we're going
            # to count all intermediary frames
            motion_count += (np.array(motion) > cls.MOTION_MIN).sum()

        return motion_count > cls.MIN_MOTION_LENGTH


class SlidePullManager(threading.Thread):

    def __init__(self, max_length=5, num_samples=4):
        super(SlidePullManager, self).__init__()
        self.max_length = max_length
        self.num_samples = num_samples
        self._current = list()
        self._processing_queue = queue.Queue()
        self.previous_pulls = deque(maxlen=num_samples)
        self.running = True
        self.run_until_empty = False
        self.last_pull = None
        self.motion_threshold = 5  # 10
        self._run_analysis = True
        self.vfp = VideoFrameProcessor()

    def _get_indexes(self, slide_pull):
        primed_idx = None
        recovery_idx = None
        for i, n in enumerate(slide_pull):
            if n.primed_idx is not None:
                primed_idx = (i, n.primed_idx)
            if n.recovery_idx is not None:
                recovery_idx = (i, n.recovery_idx)
        return primed_idx, recovery_idx

    def _find_drop(self, quick_array):
        pass

    def _recalibrate_indexes(self, slide_pull, quick_arr, slow_arr, motion_data):
        primed = False
        primed_cntr = 0
        recovery_cntr = 0
        recal_primed_idx = None
        recal_recovery_idx = None
        slide_idx = None

        for i in range(len(quick_arr)):
            quick = quick_arr[i]
            slow = slow_arr[i]
            # motion = self.vfp.process_motion_array(slide_pull[i].motion_data)
            motion_greater_than_thresh = [x > self.motion_threshold for x in motion_data[i]]
            calibration_found = [slow[j][0] for j in range(len(quick))]
                                 #[quick[j] or slow[j] for j in range(len(quick))]
            slide_drop_arr = [quick[j][1] or slow[j][1] for j in range(len(quick))]

            primed, p_idx, r_idx, primed_cntr, recovery_cntr = self.vfp.search_for_events(
                primed, calibration_found, motion_greater_than_thresh, primed_cntr, recovery_cntr
            )

            if p_idx is not None:
                log.debug("Prime idx: {}.{}".format(i, p_idx))
                recal_primed_idx = (i, p_idx)

            if True in slide_drop_arr:
                s_idx = slide_drop_arr.index(True)
                slide_idx = (i, s_idx)
                log.debug("Slide idx: {}.{}".format(i, s_idx))

            if r_idx is not None:
                log.debug("Recovery idx: {}.{}".format(i, r_idx))
                recal_recovery_idx = (i, r_idx)
                break

        if recal_primed_idx is None:
            log.info("False Pull found, aborting!")
            return None, None, None, None

        if recal_recovery_idx is None:
            log.info("No recovery found!, aborting!")
            return None, None, None, None

        if slide_idx is None:
            log.info("No slide drop found!")
            return None, None, None, None

        out = list()
        offset = recal_primed_idx[0]
        for i in range(offset, recal_recovery_idx[0]+1):
            pull = slide_pull[i]
            pull.primed_idx = None
            pull.recovery_idx = None
            if recal_primed_idx[0] == i:
                pull.primed_idx = recal_primed_idx[1]
            if recal_recovery_idx[0] == i:
                pull.recovery_idx = recal_recovery_idx[1]
            if slide_idx and slide_idx[0] == i:
                pull.slide_idx = slide_idx[1]
            else:
                pull.slide_idx = None

            out.append(pull)

        primed_idx = (recal_primed_idx[0]-offset, recal_primed_idx[1])
        recovery_idx = (recal_recovery_idx[0]-offset, recal_recovery_idx[1])
        slide_idx = (slide_idx[0]-offset, slide_idx[1])

        return out, primed_idx, recovery_idx, slide_idx

    def single_pass_run(self, slide_pull):
        primed_idx, recovery_idx = self._get_indexes(slide_pull)
        quick_arr, slow_arr, motion_data = self.vfp.generate_new_calibrations(slide_pull, slide_pull[recovery_idx[0]].calibration_state)
        recal_slide_pull, recal_primed_idx, recal_recovery_idx, slide_idx = self._recalibrate_indexes(slide_pull, quick_arr, slow_arr, motion_data)

        valid = True
        if recal_primed_idx is None:
            log.info("Invalid Slide Pull, removing")
            valid = False

        if slide_idx is None:
            log.info("Removing due to no slide drop found")
            valid = False

        log.info("Valid: {} Bounds Old: {}:{} len {} New: {}:{} len {}".format(
            valid, primed_idx, recovery_idx, len(slide_pull),
            recal_primed_idx, recal_recovery_idx, len(recal_slide_pull) if recal_slide_pull is not None else None
        ))

        return valid, recal_slide_pull, slide_idx

    def run(self):
        while self.running:
            try:
                slide_pull = self._processing_queue.get(timeout=1)
            except queue.Empty:
                if self.run_until_empty:
                    self.running = False
                continue

            log.info("Got a slide pull: {}".format([(x.primed_idx, x.recovery_idx) for x in slide_pull]))
            self.last_pull = slide_pull
            ### DEBUG
            # continue
            if not self._run_analysis:
                continue

            primed_idx, recovery_idx = self._get_indexes(slide_pull)
            quick_arr, slow_arr, motion_data = self.vfp.generate_new_calibrations(slide_pull, slide_pull[recovery_idx[0]].calibration_state)
            recal_slide_pull, recal_primed_idx, recal_recovery_idx, slide_idx = self._recalibrate_indexes(slide_pull, quick_arr, slow_arr, motion_data)

            if recal_primed_idx is None:
                log.info("Invalid Slide Pull, removing")
                continue

            if slide_idx is None:
                log.info("Removing due to no slide drop found")
                continue

            log.info("Bounds Old: {}:{} len {} New: {}:{} len {}".format(
                primed_idx, recovery_idx, len(slide_pull),
                recal_primed_idx, recal_recovery_idx, len(recal_slide_pull)
            ))

            self.previous_pulls.append((recal_slide_pull, slide_idx))

        log.info("Stopping SlidePullManager")

    def add_sample(self, grouping):
        if grouping.recovery_idx is not None:
            self._current.append(grouping)
            current_data = self._current  # This is needed due to pass by reference

            # Basic sanity filtering
            if VideoFrameProcessor.verify_has_motion(current_data):
                self._processing_queue.put(current_data)
            else:
                log.info("Filtered last drop due to motion or length violation")

            self._current = list()
        elif len(self._current) >= self.max_length:
            if grouping.primed_idx is not None:
                self._current[-1] = grouping
                log.info("Found new primed idx, removing last")
            else:
                log.info("Discarding frames due to long intermediate")
        else:
            self._current.append(grouping)

    def is_processing_queue_empty(self):
        return self._processing_queue.empty()

    def get_processing_queue_serialized(self):
        data = self._processing_queue.get()
        start_time = time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime(data[0].start_time)
        )

        return [x.serialize() for x in data], start_time


class ImageGrouping():
    def __init__(self, start_time, jpeg_data, motion_data, calibration_data,
                 primed_idx=None, recovery_idx=None, calibration_state=None):
        self.start_time = start_time
        self.jpeg_data = jpeg_data
        self.motion_data = motion_data
        self.calibration_data = calibration_data
        self.primed_idx = primed_idx
        self.recovery_idx = recovery_idx
        self.slide_idx = None
        self.calibration_state = calibration_state

    def serialize(self):
        num_entries = len(self.jpeg_data)
        calibration_data_out = b''.join([bytes([x[0]]) + x[1].tobytes('C') for x in self.calibration_data])

        if self.calibration_state is not None:
            calibration_state = CalibrationDetector.compress_serialized(self.calibration_state)
        else:
            calibration_state = None

        out = {
            'ver': 1,
            'start_time': self.start_time,
            'num_entries': num_entries,
            'jpeg_data': [binascii.b2a_base64(x).decode('ascii') for x in self.jpeg_data],
            'motion_data': self.motion_data,
            'calibration_data': binascii.b2a_base64(calibration_data_out).decode('ascii'),
            'primed_idx': self.primed_idx,
            'recovery_idx': self.recovery_idx,
            'slide_idx': self.slide_idx,
            'calibration_state': calibration_state
        }

        #jpeg_data_len = struct.pack('<' + 'I'*len(num_entries), *[len(x) for x in self.jpeg_data])
        #jpeg_data = b''.join(self.jpeg_data)
        #motion_data = struct.pack('f'*num_entries, *self.motion_data)

        return out

    @classmethod
    def deserialize(cls, inp):
        assert inp['ver'] == 1

        num_entries = inp['num_entries']
        jpeg_data = [binascii.a2b_base64(x) for x in inp['jpeg_data']]
        motion_data = inp['motion_data']
        cd_binary = binascii.a2b_base64(inp['calibration_data'])
        calibration_data = []

        stride = 3*8 + 1
        for i in range(0, stride*num_entries, stride):
            cut = cd_binary[i:i+stride]
            found = True if cut[0] == 1 else False
            params = np.frombuffer(cut[1:])
            calibration_data.append((found, params))

        c = cls(
            start_time=inp['start_time'],
            jpeg_data=jpeg_data,
            motion_data=motion_data,
            calibration_data=calibration_data,
            primed_idx=inp['primed_idx'],
            recovery_idx=inp['recovery_idx'],
            calibration_state=CalibrationDetector.decompress_serialized(inp['calibration_state'])
        )

        return c


class ProcessingQueue(threading.Thread):
    def __init__(self, output_queue, callback, run_until_empty=False):
        super(ProcessingQueue, self).__init__()
        self.queue = output_queue
        self.running = True
        self.cd = CalibrationDetector()
        self.run_until_empty = run_until_empty
        self.primed = False
        self.callback = callback
        self._primed_cntr = 0
        self._recovery_cntr = 0
        self.last_frame = None
        self.motion_threshold = 5  # 10
        self._last_raw_frame = None

    def process_calibration(self, opencv_frames):
        res = []
        found = 0
        for i, x in enumerate(opencv_frames):
            if not self.cd.is_locked:
                if i % 5 == 0:
                    similar, calib = self.cd.process_frame(x)
                else:
                    similar, calib = False, [0, 0, 0]
            else:
                similar, calib = self.cd.quick_check_frame(x)
                if not similar and i == 0:
                    log.debug("First Frame in group failed similar check, testing again")
                    similar, calib = self.cd.process_frame(x)
            res.append((similar, calib))
            found += 1 if similar else 0

        return res, found

    def run(self):
        while self.running:
            try:
                ret = self.queue.get(timeout=1)
            except queue.Empty:
                if self.run_until_empty:
                    self.running = False
                continue

            self.last_frame = ret.jpeg[0]

            # For motion detection, we need a previous frame, on frame 1 just use the first frame twice
            if self._last_raw_frame is None:
                self._last_raw_frame = ret.raw[0]
                continue

            start = time.time()

            motion_res = VideoFrameProcessor.generate_motion_parameters(
                self._last_raw_frame, ret.raw
            )

            self._last_raw_frame = ret.raw[-1].copy()
            motion_greater_than_thresh = [x > self.motion_threshold for x in motion_res]
            num_with_motion = sum(motion_greater_than_thresh)
            calib_parameters, total_calib_found = self.process_calibration(ret.raw)
            primed, primed_idx, recovery_idx, primed_cntr, recovery_cntr = VideoFrameProcessor.search_for_events(
                self.primed,
                [x[0] for x in calib_parameters],
                motion_greater_than_thresh,
                self._primed_cntr,
                self._recovery_cntr
            )
            self.primed = primed
            self._primed_cntr = primed_cntr
            self._recovery_cntr = recovery_cntr
            end = time.time()

            if primed_idx is not None or recovery_idx is not None:
                if recovery_idx is not None:
                    state = self.cd.serialize_state()
                else:
                    state = None
                ig = ImageGrouping(ret.start_time, ret.jpeg, motion_res, calib_parameters, primed_idx, recovery_idx, state)
                log.debug("Adding Sample: P{} R{}".format(primed_idx, recovery_idx))
                self.callback(ig)

            log.debug("Got Data: Time: {:.2f}, Found {}/{} Motion {}/{} in {:.5f} seconds Avg {:.5f} per frame. P{} PI{} RI{} Size: J{} R{} J{} R{}".format(
                ret.start_time,
                total_calib_found, len(ret.raw),
                num_with_motion, len(motion_res),
                end-start,
                (end-start)/len(ret.raw),
                self.primed,
                primed_idx,
                recovery_idx,
                sum(map(len, ret.jpeg)),
                sum(map(lambda x: x.size, ret.raw)),
                ret.jpeg_meta[-1],
                ret.raw_meta[-1]
            ))
        log.info("Processing Queue Thread Stopping")
