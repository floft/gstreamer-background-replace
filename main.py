#!/usr/bin/env python3
"""
GStreamer Background Replace

Zoom on Linux has poor virtual background support (unless you have a solid
backdrop), so instead route through GStreamer/Python and replace the background
using a pre-trained semantic segmentation neural net.

Useful docs:
https://gstreamer.freedesktop.org/documentation/gstreamer/gi-index.html?gi-language=python
https://lazka.github.io/pgi-docs/Gst-1.0/index.html
"""
import os
import time
import tarfile
import threading
import numpy as np
import tensorflow as tf
import urllib.request

from absl import app
from absl import flags
from collections import deque
from PIL import Image

# Gstreamer
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
gi.require_version("GObject", "2.0")
from gi.repository import GLib, GObject, Gst

FLAGS = flags.FLAGS

flags.DEFINE_string("filename", "", "Input filename, if specified will read this rather than the V4L2 input_device")
flags.DEFINE_string("input_device", "", "Input V4L2 device, e.g. /dev/video0")
flags.DEFINE_string("output_device", "", "Output V4L2 device from the v4l2loopback driver, if not specified then outputs with GStreamer")
flags.DEFINE_string("background", "", "Background image, if desired (otherwise it's just a black background)")
flags.DEFINE_integer("output_width", 1280, "Desired output width (i.e. probably the webcam output dimensions)")
flags.DEFINE_integer("output_height", 720, "Desired output height (i.e. probably the webcam output dimensions)")
flags.DEFINE_integer("crop_width", 0, "Crop/pad to this if smaller/larger than webcam output, then scaled to output_width (0 = no crop)")
flags.DEFINE_integer("crop_height", 0, "Crop/pad to this if smaller/larger than webcam output, then scaled to output_height (0 = no crop)")
flags.DEFINE_integer("framerate", 30, "Desired output framerate (not sure this matters)")
flags.DEFINE_float("color_temp", 0.433333, "Whitebalance -- adjust color temperature (0-1)")
flags.DEFINE_float("green_tint", 0.133333, "Whitebalance -- adjust green tint (0-1)")
flags.DEFINE_float("vignette_size", 1.0, "Vignette size (size of unaffected circle in center) -- 0 = everything black, 1 = no vignette")
flags.DEFINE_float("vignette_soft", 1.0, "Vignette soft -- 0 = completely hard edge, 1 = no vignette")
flags.DEFINE_boolean("low_latency", True, "Whether to have low-latency, i.e. audio matches video more closely")
flags.DEFINE_boolean("blur", False, "Whether to blur the foreground/background mask -- slower")
flags.DEFINE_boolean("debug", False, "Whether to output FPS information")
flags.DEFINE_boolean("lite", True, "Whether to use the TF lite model (faster)")


def load_model(model_name="mobilenetv2_coco_voctrainaug"):
    lite = model_name == "lite"

    prefix = "https://storage.googleapis.com/download.tensorflow.org/models/"
    urls = {
        "mobilenetv2_coco_voctrainaug":
            "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz",
        "mobilenetv2_coco_voctrainval":
            "deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz",
        "xception_coco_voctrainaug":
            "deeplabv3_pascal_train_aug_2018_01_04.tar.gz",
        "xception_coco_voctrainval":
            "deeplabv3_pascal_trainval_2018_01_04.tar.gz",
        "lite":
            "tflite/gpu/deeplabv3_257_mv_gpu.tflite",
    }

    if lite:
        download_path = "deeplab_model.tflite"
    else:
        download_path = "deeplab_model.tar.gz"

    if not os.path.exists(download_path):
        print("Downloading model")
        url = prefix + urls[model_name]
        urllib.request.urlretrieve(url, download_path)

    print("Loading model")
    if lite:
        model = DeepLabModelLite(download_path)
    else:
        # Convert to TF Lite so it's faster...
        model = DeepLabModelConvertToLite(download_path)
        # model = DeepLabModel(download_path)

    print("Model loaded")

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

    (success, width) = caps_struct.get_int("width")
    if not success:
        return False, (0, 0)

    (success, height) = caps_struct.get_int("height")
    if not success:
        return False, (0, 0)

    return True, (width, height)


class DeepLabModel:
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = "ImageTensor:0"
    OUTPUT_TENSOR_NAME = "SemanticPredictions:0"
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError("Cannot find inference graph in tar archive.")

        with self.graph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name="")

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def calc_target_size(self, size):
        width, height = size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        return target_size

    def run(self, image):
        """ Runs inference on a single image.

        Args: numpy array already at the correct size
        Returns: Segmentation map of `resized_image`.
        """
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [image]})
        seg_map = batch_seg_map[0]
        return seg_map

    def true_model_shape(self, width, height):
        model_width = int(DeepLabModel.INPUT_SIZE)

        # Need this otherwise we get reshape errors... GStreamer videoscale
        # doesn't scale to any arbitrary value, I suspect.
        model_width = round_to_even(model_width)

        model_height = int(DeepLabModel.INPUT_SIZE / (width / height))

        return model_width, model_height


class DeepLabModelLite:
    """Class to load deeplab model and run inference -- TF Lite version

    See: https://www.tensorflow.org/lite/guide/inference
    Model from: https://www.tensorflow.org/lite/models/segmentation/overview
    """
    def __init__(self, filename=None, model_content=None):
        """ Creates and loads pretrained deeplab model. """
        # Load TFLite model and allocate tensors.
        if model_content is None:
            assert filename is not None, "must pass tflite filename"
            self.interpreter = tf.lite.Interpreter(model_path=filename)
        else:
            self.interpreter = tf.lite.Interpreter(model_content=model_content)

        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get shapes and I/O
        self.input_shape = self.input_details[0]["shape"]
        self.input_tensor = self.input_details[0]["index"]
        self.output_tensor = self.output_details[0]["index"]

        # Whether it's a float or quantized model
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True
            print("Floating-point model detected")
        else:
            self.floating_model = False
            print("Quantized model detected")

    @tf.function
    def pre_process(self, image):
        # Convert to float
        if self.floating_model:
            image = (tf.cast(image, dtype=tf.float32) - 127.5) / 127.5

        # Expand dimensions since the model expects images to have shape:
        #   [1, height, width, 3]
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        else:
            assert len(image.shape) == 4, "should have HWC or NHWC (where N=1)"

        return image

    @tf.function
    def post_process(self, batch):
        result = batch[0]

        # This outputs 257x257x21 whereas the non-TF lite one outputs an
        # int I think? So, convert to the values being the class label not
        # a softmax sort of thing.
        #
        # Also, convert/round to int8 (# classes i.e. 21 << 256 values)
        if len(result.shape) == 3:
            result = tf.cast(tf.argmax(result, axis=-1), dtype=tf.uint8)
        # Otherwise... it's already done this if quantized???

        return result

    def run(self, image):
        """ Runs inference on a single image.

        Args: a numpy array already at the correct size
        Returns: seg_map: Segmentation map of `resized_image`.
        """
        image = self.pre_process(image)

        # TF Lite set_tensor requires it be a np.float32, not tf.float32
        image = image.numpy()

        # Run data through model
        self.interpreter.set_tensor(self.input_tensor, image)
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        batch = self.interpreter.get_tensor(self.output_tensor)

        return self.post_process(batch)

    def true_model_shape(self, width, height):
        """ The actual shape to run through the model """
        # NxHxWxC
        # shape: 1, 257, 257, 3
        input_width = self.input_shape[2]
        input_height = self.input_shape[1]
        return input_width, input_height


class DeepLabModelConvertToLite(DeepLabModelLite):
    """ Convert model to TF Lite then run"""

    INPUT_TENSOR_NAME = "ImageTensor"
    OUTPUT_TENSOR_NAME = "SemanticPredictions"
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        # Load the full model
        frozen_pb_file = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                frozen_pb_file = tar_info.name
                break

        # Conversion requires an actual files
        tar_file.extractall()
        tar_file.close()

        if frozen_pb_file is None:
            raise RuntimeError("Cannot find inference graph in tar archive.")

        # See: https://stackoverflow.com/a/59752911
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            frozen_pb_file,
            input_shapes={self.INPUT_TENSOR_NAME:
                [1, self.INPUT_SIZE, self.INPUT_SIZE, 3]},
            input_arrays=[self.INPUT_TENSOR_NAME],
            output_arrays=[self.OUTPUT_TENSOR_NAME])

        converter.inference_type = tf.uint8
        converter.quantized_input_stats = {self.INPUT_TENSOR_NAME: (127.5, 127.5)}

        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8

        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]

        model_content = converter.convert()

        # Then use the TF Lite code
        super().__init__(model_content=model_content)


def round_to_even(val):
    return 2 * round(val / 2)


def calc_gaussian_blur_kernel(num_channels=3, kernel_size=7, sigma=5):
    """ From: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319 """
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(num_channels, kernel_size, sigma)
    return gaussian_kernel[..., tf.newaxis]


def gaussian_blur(img, gaussian_kernel):
    # Expand to have batch dimension
    img = tf.expand_dims(img, axis=0)

    # Do the blur
    blurred = tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
        padding="SAME")

    # Get rid of batch dimension
    return blurred[0]


class BackgroundReplacement:
    """ Wrap model to calculate FPS and run in GStreamer

    See: https://github.com/floft/vision-landing/blob/master/object_detector.py
    """
    def __init__(self, average_fps_frames=30, debug=False, lite=False,
            crop_width=None, crop_height=None,
            output_width=1280, output_height=720, framerate=30,
            green_tint=0.133333, color_temp=0.433333, filename=None,
            vignette_size=1, vignette_soft=1, background=None, blur=False,
            device="/dev/video0", v4l2outputdevice=None, low_latency=False):
        self.debug = debug
        self.lite = lite
        self.blur = blur
        self.exiting = False

        assert filename is not None or device is not None, \
            "must pass either filename= or device="

        if lite:
            self.model = load_model("lite")
        else:
            self.model = load_model()

        if crop_width is None or crop_width == 0:
            crop_width = output_width
        if crop_height is None or crop_height == 0:
            crop_height = output_height

        # GStreamer can't resize to all shapes exactly, so we may pad it
        # a couple pixels.
        self.true_model_width, self.true_model_height = self.model.true_model_shape(
            crop_width, crop_height)

        # Pre-compute for later
        self.gaussian_kernel = calc_gaussian_blur_kernel(num_channels=1)
        self.alpha_channel = tf.cast(tf.ones((crop_height, crop_width, 1))*255, dtype=tf.uint8)

        # Get the background, if specified
        self.background_image = None

        if background is not None and background != "":
            # Resize/crop/pad to forground size
            self.background_image = np.array(Image.open(background))
            self.background_image = tf.cast(tf.image.resize_with_crop_or_pad(
                self.background_image, crop_height, crop_width),
                dtype=tf.float32).numpy()

            # Drop the alpha channel, e.g. if from a png image
            if self.background_image.shape[-1] == 4:
                self.background_image = self.background_image[:, :, :-1]

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
        if filename is not None and filename != "":
            launch_str = "filesrc location=\"" + filename + "\" ! decodebin"
            live = False
        else:
            launch_str = "v4l2src device=\"" + device + "\" ! jpegdec"
            live = True

        launch_str += " ! videobox autocrop=true " \
            "! video/x-raw," \
            "width=" + str(crop_width) + "," \
            "height=" + str(crop_height) + " " \
            "! videoconvert ! videoscale " \
            "! video/x-raw,format=RGBA " \
            "! frei0r-filter-white-balance green-tint=" + str(green_tint) + " " \
            "! frei0r-filter-white-balance--lms-space- " \
            "color-temperature=" + str(color_temp) + " " \
            "! frei0r-filter-vignette clearcenter=" + str(vignette_size) + " " \
            "soft=" + str(vignette_soft) + " " \
            "! appsink name=appsink"
        self.load_pipe = Gst.parse_launch(launch_str)

        self.appsink = self.load_pipe.get_by_name("appsink")
        self.appsink.set_property("qos", True)
        self.appsink.set_property("sync", True)
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
            "width=" + str(crop_width) + "," \
            "height=" + str(crop_height) + " " \
            "! videoconvert ! videoscale " \
            "! video/x-raw,format=YUY2," \
            "width=" + str(output_width) + "," \
            "height=" + str(output_height) + "," \
            "framerate=" + str(framerate) + "/1," \
            "interlace-mode=progressive ! " \

        if v4l2outputdevice is not None and v4l2outputdevice != "":
            launch_str += "v4l2sink device=\"" + v4l2outputdevice + "\""
        else:
            launch_str += "videoconvert ! autovideosink"

        self.display_pipe = Gst.parse_launch(launch_str)
        self.appsrc = self.display_pipe.get_by_name("appsrc")
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("duration", 1/framerate)

        if live and low_latency:
            self.appsrc.set_property("block", False)
            self.appsrc.set_property("is_live", True)
            # So... there will be some lag I think (since we're applying a
            # slightly later timestamp), but this at least makes it not drop
            # 80% of the frames.
            self.appsrc.set_property("do_timestamp", True)

        # Set the second pipeline's clock to the first's
        self.display_pipe.use_clock(self.load_pipe.get_pipeline_clock())

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
            self.gst_next_frame(frame, buf.dts, buf.duration, buf.offset,
                buf.offset_end, buf.pts)
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

    @tf.function
    def segment_pre_process(self, np_image):
        # Drop alpha channel
        np_image = np_image[:, :, :-1]

        # Resize to shape that we can input to the model, and do it fast
        np_image_small = tf.image.resize(np_image,
            (self.true_model_height, self.true_model_width),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return np_image, np_image_small

    @tf.function
    def segment_post_process(self, np_image, seg_map):
        # Zero the background for now
        mask = tf.cast(seg_map != 0, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=-1)  # it's one channel

        # Blur
        if self.blur:
            mask = gaussian_blur(mask, self.gaussian_kernel)

        # Scale up alpha to match image shape
        output_height, output_width, _ = np_image.shape
        alpha = tf.image.resize(mask, (output_height, output_width),
            method=tf.image.ResizeMethod.BILINEAR, antialias=True)

        # Replace background
        if self.background_image is None:
            np_image = tf.cast(np_image, dtype=tf.float32) * alpha
        else:
            np_image = tf.cast(np_image, dtype=tf.float32) * alpha \
                + (1 - alpha) * self.background_image

        # Clip to 0-255 and cast to uint
        np_image = tf.cast(tf.clip_by_value(np_image, 0, 255), dtype=tf.uint8)

        # Add in alpha channel
        return tf.concat([np_image, self.alpha_channel], axis=-1)

    def segment(self, np_image):
        # Pre-process
        np_image, np_image_small = self.segment_pre_process(np_image)

        # Run through model
        seg_map = self.model.run(np_image_small)

        # Post-process
        np_image = self.segment_post_process(np_image, seg_map)

        # Output is expected to be numpy array
        return np_image.numpy()

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

    def gst_next_frame(self, frame, dts, duration, offset, offset_end, pts):
        """ When we have a new numpy array RGB image, push it to GStreamer """
        data = frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)

        # Copy time information from when the webcam frame was captured
        buf.dts = dts
        buf.duration = duration
        buf.offset = offset
        buf.offset_end = offset_end
        buf.pts = pts

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


def main(argv):
    otherargs = {
        "output_width": FLAGS.output_width,
        "output_height": FLAGS.output_height,
        "crop_width": FLAGS.crop_width,
        "crop_height": FLAGS.crop_height,
        "framerate": FLAGS.framerate,
        "green_tint": FLAGS.green_tint,
        "color_temp": FLAGS.color_temp,
        "vignette_size": FLAGS.vignette_size,
        "vignette_soft": FLAGS.vignette_soft,
        "blur": FLAGS.blur,
        "low_latency": FLAGS.low_latency,
        "background": FLAGS.background,
        "debug": FLAGS.debug,
        "lite": FLAGS.lite,
    }

    if FLAGS.filename != "":
        # On video
        br = BackgroundReplacement(filename=FLAGS.filename, **otherargs)
    else:
        # On live webcam images and output to virtual V4L2 device (if specified)
        # or if not then GStreamer output window
        br = BackgroundReplacement(device=FLAGS.input_device,
            v4l2outputdevice=FLAGS.output_device, **otherargs)

    br.run()


if __name__ == "__main__":
    app.run(main)
