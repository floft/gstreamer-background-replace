#!/usr/bin/env python3
"""
Run replacement on live images in GStreamer

I went with the TensorFlow version since I think the model is smaller and
probably runs faster...
"""
import os
import time
import tarfile
import threading

import numpy as np
from collections import deque
from PIL import Image
from six.moves import urllib
from scipy.ndimage.filters import gaussian_filter

# Gstreamer
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
from gi.repository import GLib, GObject, Gst

# This code was for TF 1.x and I think the model is trained on that too?
# https://www.tensorflow.org/guide/migrate
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_model(model_name='mobilenetv2_coco_voctrainaug'):
    lite = model_name == "lite"

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        'lite':
            'tflite/gpu/deeplabv3_257_mv_gpu.tflite',
    }

    if lite:
        download_path = 'deeplab_model.tflite'
    else:
        download_path = 'deeplab_model.tar.gz'

    if not os.path.exists(download_path):
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[model_name],
                        download_path)
        print('download completed! loading DeepLab model...')

    if lite:
        model = DeepLabModelLite(download_path)
    else:
        model = DeepLabModel(download_path)

    print('model loaded successfully!')

    return model


def get_buffer_size(caps):
    """
    Returns width, height of buffer from caps
    Taken from: http://lifestyletransfer.com/how-to-get-buffer-width-height-from-gstreamer-caps/
    :param caps: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Caps.html
    :type caps: Gst.Caps
    :rtype: bool, (int, int)
    """
    caps_struct = caps.get_structure(0)

    (success, width) = caps_struct.get_int('width')
    if not success:
        return False, (0, 0)

    (success, height) = caps_struct.get_int('height')
    if not success:
        return False, (0, 0)

    return True, (width, height)


class DeepLabModel:
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def calc_target_size(self, size):
        width, height = size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        return target_size

    def run(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.
        (or if resize=False, then a numpy array already at the correct size,
        for when we resize within GStreamer)

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        # if resize:
        #     target_size = self.calc_target_size(image.size)
        #     resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        #     image = np.asarray(resized_image)
        # else:
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [image]})
        seg_map = batch_seg_map[0]
        return seg_map

    def gst_shape(self, width, height):
        model_width = int(DeepLabModel.INPUT_SIZE)

        # Need this otherwise we get reshape errors... GStreamer videoscale
        # doesn't scale to any arbitrary value, I suspect.
        model_width = round_to_even(model_width)

        model_height = int(DeepLabModel.INPUT_SIZE / (width / height))

        return model_width, model_height

    def true_model_shape(self, width, height):
        # Works?
        return self.gst_shape(width, height)


class DeepLabModelLite:
    """Class to load deeplab model and run inference -- TF Lite version

    See: https://www.tensorflow.org/lite/guide/inference
    Model from: https://www.tensorflow.org/lite/models/segmentation/overview
    """
    def __init__(self, filename):
        """ Creates and loads pretrained deeplab model. """
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=filename)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get shapes and I/O
        self.input_shape = self.input_details[0]['shape']
        self.input_tensor = self.input_details[0]['index']
        self.output_tensor = self.output_details[0]['index']

    def run(self, image):
        """ Runs inference on a single image.

        Args: a numpy array already at the correct size
        Returns: seg_map: Segmentation map of `resized_image`.
        """
        # Convert to float
        image = (np.float32(image) - 127.5) / 127.5

        # Expand dimensions since the model expects images to have shape:
        #   [1, height, width, 3]
        image = np.expand_dims(image, axis=0)

        # Run data through model
        self.interpreter.set_tensor(self.input_tensor, image)
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        batch = self.interpreter.get_tensor(self.output_tensor)
        result = batch[0]

        # This outputs 257x257x21 whereas the non-TF lite one outputs an
        # int I think? So, convert to the values being the class label not
        # a softmax sort of thing.
        #
        # Also, convert/round to int8 (# classes i.e. 21 << 256 values)
        result = np.argmax(result, axis=-1).astype(np.uint8)

        return result

    def gst_shape(self, width, height):
        """ Different than DeepLabModel """
        # NxHxWxC
        # shape: 1, 257, 257, 3
        input_width = self.input_shape[2]
        # input_height = self.input_shape[1]

        # Calculate
        model_width = int(input_width)

        # Need this otherwise we get reshape errors... GStreamer videoscale
        # doesn't scale to any arbitrary value, I suspect.
        model_width = round_to_even(model_width)

        # model_height = int(input_height / (width / height))

        # return model_width, model_height

        # Small, so do a square?
        return model_width, model_width

    def true_model_shape(self, width, height):
        """ The actual shape to run through the model """
        # NxHxWxC
        # shape: 1, 257, 257, 3
        input_width = self.input_shape[2]
        input_height = self.input_shape[1]
        return input_width, input_height


def round_to_even(val):
    return 2 * round(val / 2)


class BackgroundReplacement:
    """ Wrap model to calculate FPS and run in GStreamer

    See: https://github.com/floft/vision-landing/blob/master/object_detector.py
    """
    def __init__(self, average_fps_frames=30, debug=False, lite=False,
            gst_width=854, gst_height=480, gst_framerate=30,
            gst_display_width=854, green_tint=0.04, color_temp=0.7,
            crop_top=120, crop_bottom=120, crop_left=213, crop_right=213,
            filename=None, device=None, v4l2outputdevice=None):
        self.debug = debug
        self.lite = lite
        self.exiting = False

        assert filename is not None or device is not None, \
            "must pass either filename= or device="

        if lite:
            self.model = load_model("lite")
        else:
            self.model = load_model()

        # GStreamer can't resize to all shapes exactly, so we may pad it
        # a couple pixels.
        self.gst_width = gst_width
        self.gst_height = gst_height
        self.gst_model_width, self.gst_model_height = self.model.gst_shape(
            gst_width, gst_height)
        self.true_model_width, self.true_model_height = self.model.true_model_shape(
            gst_width, gst_height)

        # compute average FPS over # of frames
        self.fps = deque(maxlen=average_fps_frames)

        # compute streaming FPS (how fast frames are arriving from camera
        # and we're able to process them, i.e. this is the actual FPS)
        self.stream_fps = deque(maxlen=average_fps_frames)
        self.process_end_last = 0

        # Run GStreamer in separate thread
        self.t_gst_load = threading.Thread(target=self.gst_run_load)
        self.t_gst_display = threading.Thread(target=self.gst_run_display)
        Gst.init(None)

        #
        # Get the input frames
        #
        if filename is not None:
            launch_str = "filesrc location=\"" + filename + "\" ! decodebin"
        else:
            launch_str = "v4l2src device=\"" + device + "\" " \
                "! jpegdec ! videocrop top=" + str(crop_top) + " " \
                "left=" + str(crop_left) + " right=" + str(crop_right) + " " \
                "bottom=" + str(crop_bottom)

        launch_str += " ! videoconvert ! videoscale " \
            "! video/x-raw,format=RGBA," \
            "width=" + str(self.gst_width) + "," \
            "height=" + str(self.gst_height) + " " \
            "! frei0r-filter-white-balance green-tint=" + str(green_tint) + " " \
            "! frei0r-filter-white-balance--lms-space- " \
            "color-temperature=" + str(color_temp) + " " \
            "! frei0r-filter-vignette clearcenter=0.4 soft=0.6 " \
            "! appsink name=appsink"
        self.load_pipe = Gst.parse_launch(launch_str)

        self.appsink = self.load_pipe.get_by_name("appsink")
        self.appsink.set_property("sync", False)
        self.appsink.set_property("drop", True)
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", lambda x: self.process_frame(x))

        #
        # Process and then view the results
        #
        # Kind of weird... only green if outputs RGB, so we output RGBA, but
        # then convert to RGB, but then we can't pass "format" at the end or
        # it errors, so we end up converting multiple times.
        launch_str = "appsrc name=appsrc " \
            "! video/x-raw,format=RGBA," \
            "width=" + str(self.gst_width) + "," \
            "height=" + str(self.gst_height) + "," \
            "framerate=" + str(gst_framerate) + "/1 " \
            "! videoconvert ! videoscale " \
            "! video/x-raw,format=YUY2," \
            "width=" + str(self.gst_width) + "," \
            "height=" + str(self.gst_height)

        if v4l2outputdevice is not None:
            launch_str += ",interlace-mode=progressive " \
                "! v4l2sink device=\"" + v4l2outputdevice + "\""
        else:
            launch_str += " ! videoconvert ! autovideosink"

        self.display_pipe = Gst.parse_launch(launch_str)
        self.appsrc = self.display_pipe.get_by_name("appsrc")

        # Event loop
        self.load_loop = GLib.MainLoop()
        self.display_loop = GLib.MainLoop()

        # Get error messages or end of stream on bus
        self._monitor_errors(self.load_pipe, self.load_loop)
        self._monitor_errors(self.display_pipe, self.display_loop)

    def _monitor_errors(self, pipe, loop):
        bus = pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.gst_bus_call, loop)

    def avg_fps(self):
        """ Return average FPS over last so many frames (specified in constructor) """
        return sum(list(self.fps))/len(self.fps)

    def avg_stream_fps(self):
        """ Return average streaming FPS over last so many frames (specified in constructor) """
        return sum(list(self.stream_fps))/len(self.stream_fps)

    def process_frame(self, frame):
        """ Get image from the appsink, process, then push to appsrc """
        # Get frame
        sample = self.appsink.emit("pull-sample")
        got_caps, (width, height) = get_buffer_size(sample.get_caps())
        assert got_caps, "Could not get width/height from buffer!"

        # See: https://github.com/TheImagingSource/tiscamera/blob/master/examples/python/opencv.py
        buf = sample.get_buffer()

        try:
            _, mapinfo = buf.map(Gst.MapFlags.READ)
            # Create a numpy array from the data
            frame = np.asarray(bytearray(mapinfo.data), dtype=np.uint8)
            # Give the array the correct dimensions of the video image
            # Note: 4 channels are R, G, B, A
            frame = frame.reshape((height, width, 4))

            # Process it
            frame = self.process(frame)

            # Then output the processed frame
            self.gst_next_frame(frame)
        finally:
            buf.unmap(mapinfo)

        return False

    def pad_to(self, data, desired_width, desired_height):
        """
        Pad to the right width/height

        shape: (height, width, channels) -> (desired_height, desired_width, channels)
        """
        assert len(data.shape) == 3
        current_height = data.shape[0]
        current_width = data.shape[1]

        assert current_height <= desired_height, \
            "Cannot shrink size by padding, current height " \
            + str(current_height) + " vs. desired height " \
            + str(desired_height)
        assert current_width <= desired_width, \
            "Cannot shrink size by padding, current width " \
            + str(current_width) + " vs. desired width " \
            + str(desired_width)

        return np.pad(data, [(0, desired_height - current_height),
                (0, desired_width - current_width),
                (0, 0)],
            mode="constant", constant_values=0)

    def resize(self, np_array, width, height):
        # https://docs.scipy.org/doc/scipy-1.2.0/reference/generated/scipy.misc.imresize.html
        return np.array(Image.fromarray(np_array).resize((width, height)))

    def segment(self, np_image):
        # Drop alpha channel
        np_image = np_image[:, :, :-1]

        # Shrink
        np_image_small = self.resize(np_image,
            self.true_model_width, self.true_model_height)

        # Pad to correct shape
        np_image_small = self.pad_to(
            np_image_small, self.true_model_width, self.true_model_height)

        # Run through model
        seg_map = self.model.run(np_image_small)

        # Zero the background for now
        mask = (seg_map != 0).astype(np.float32)  # 0: background, so everything else

        # Change alpha, but not totally black
        mask = gaussian_filter(mask, sigma=2)

        # New shape: (height, width, 3)
        alpha = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

        # # Scale up alpha to match image shape
        output_height, output_width, _ = np_image.shape
        alpha_large = self.resize((alpha * 255).astype(np.uint8), output_width, output_height)
        alpha_large = (alpha_large / 255).astype(np.float32)

        # Crop to the output shape (if we padded by 1px for example)
        # np_image = np_image[0:self.gst_model_height, 0:self.gst_model_height]

        # Multiply
        np_image = (np_image.astype(np.float32) * alpha_large).astype(np.uint8)

        # Add in alpha channel, since otherwise we get completely green output?
        # https://stackoverflow.com/a/39643014
        np_image = np.dstack(
            (np_image, np.ones((output_height, output_width))*255))

        # Output is expected to be uint8
        return np_image.astype(np.uint8)

    def process(self, *args, **kwargs):
        if self.debug:
            # Start timer
            fps = time.time()

        results = self.segment(*args, **kwargs)

        if self.debug:
            now = time.time()

            # End timer
            fps = 1/(now - fps)
            self.fps.append(fps)

            # Streaming FPS
            stream_fps = 1/(now - self.process_end_last)
            self.stream_fps.append(stream_fps)
            self.process_end_last = now

            print("Stats:",
                "Process FPS", "{:<5}".format("%.2f"%self.avg_fps()),
                "Stream FPS", "{:<5}".format("%.2f"%self.avg_stream_fps()))

        return results

    def run(self):
        self.gst_start()

        # Run till Ctrl+C
        try:
            while True:
                time.sleep(600)
        except KeyboardInterrupt:
            self.exiting = True
            self.gst_stop()

    def gst_bus_call(self, bus, message, loop):
        """ Print important messages """
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error: %s: %s" % (err, debug))
            loop.quit()
        return True

    def gst_next_frame(self, frame):
        """ When we have a new numpy array RGB image, push it to GStreamer """
        data = frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)
        self.appsrc.emit("push-buffer", buf)

    def _gst_run(self, pipe, loop):
        pipe.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        finally:
            pipe.set_state(Gst.State.NULL)

    def gst_run_load(self):
        """ This is run in a separate thread. Start, loop, and cleanup. """
        self._gst_run(self.load_pipe, self.load_loop)

    def gst_run_display(self):
        """ This is run in a separate thread. Start, loop, and cleanup. """
        self._gst_run(self.display_pipe, self.display_loop)

    def gst_start(self):
        """ If using GStreamer, start that thread """
        self.t_gst_load.start()
        self.t_gst_display.start()

    def gst_stop(self):
        """ If using GStreamer, tell it to exit and then wait """
        self.load_loop.quit()
        self.display_loop.quit()
        self.t_gst_load.join()
        self.t_gst_display.join()


if __name__ == "__main__":
    # On video
    # br = BackgroundReplacement(filename="test.avi", debug=True, lite=True)

    # On live webcam images
    # br = BackgroundReplacement(device="/dev/video2", debug=True, lite=True)

    # On live webcam images and output to virtual V4L2 device
    br = BackgroundReplacement(device="/dev/video2",
        v4l2outputdevice="/dev/video7", debug=True, lite=True)

    br.run()
