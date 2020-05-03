GStreamer Background Replace
============================

Zoom background replacement doesn't work well for me on Linux. Thus, let's use a
neural net to segment the image and replace the background. And... then just
hope it's fast enough.

Also, while we're at it, adjust the white balance, crop the too-wide-angle
GoPro view, add a slight vignette, etc.

## Installation
If you want to output this as a virtual/fake webcam on Linux, e.g. to use in
Zoom, Hangouts, Messenger, etc. then install
[v4l2loopback](https://github.com/umlaeute/v4l2loopback), for example if on Arch
with the AUR package [v4l2loopback-dkms](https://aur.archlinux.org/packages/v4l2loopback-dkms/) (though make sure you have *linux-headers* installed as well, or else dkms won't build it for your kernel).

Then, gstreamer 1.0 with good/bad plugins, TensorFlow, Numpy, SciPy, and abseil-py (absl-py).

On Arch:

    sudo pacman -S linux-headers python-{scipy,numpy,tensorflow} gst-{python,plugins{good,bad}} absl-py
    aur sync v4l2loopback-dkms

(Or, python-tensorflow-cuda, python-tensorflow-opt, ...)

## Usage

Load the kernel module:

    sudo modprobe v4l2loopback video_nr=7 exclusive_caps=1

Run the script (set the appropriate device, v4l2 output device, etc.):

    ./gstreamer_replace.py --input_device="/dev/video0" --output_device="/dev/video7"

Or, see *run.sh* for a full example.

## Result

Decent, though not that great:

![Result](https://raw.githubusercontent.com/floft/gstreamer-background-replace/master/files/zoom_gstreamer_replace.png)

Compare that with Zoom's internal replacement algorithm result:

![Zoom Background](https://raw.githubusercontent.com/floft/gstreamer-background-replace/master/files/zoom_internal_replace.png)
