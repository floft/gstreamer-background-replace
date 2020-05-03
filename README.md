GStreamer Background Replace
============================

Zoom background replacement doesn't work well for me. Thus, let's use a neural
net to segment the image and replace the background. And... then just hope it's
fast enough.

Also, while we're at it, adjust the white balance, crop the too-wide-angle
GoPro view, add a slight vignette, etc.

## Installation
If you want to output this as a virtual/fake webcam on Linux, e.g. to use in
Zoom, Hangouts, Messenger, etc. then install
[v4l2loopback](https://github.com/umlaeute/v4l2loopback), for example if on Arch
with the AUR package [v4l2loopback-dkms](https://aur.archlinux.org/packages/v4l2loopback-dkms/) (though make sure you have *linux-headers* installed as well, or else dkms won't build it for your kernel).

Then, of course, you need gstreamer 1.0 with all the various
plugins we use (included in the good and bad plugin packages).

And, install TensorFlow

On Arch:

    sudo pacman -S linux-headers python-tensorflow
    aur sync v4l2loopback-dkms

(Or, python-tensorflow-cuda, python-tensorflow-opt, ...)

## Usage

Load the kernel module:

    sudo modprobe v4l2loopback video_nr=7 exclusive_caps=1

Run the script (set the appropriate device, v4l2 output device, etc.):

    ./gstreamer_replace.py

## Result

Not that great (yet), but it's much better than Zoom's result.

![Result](https://raw.githubusercontent.com/floft/gstreamer-background-replace/master/files/screenshot.png)

Compare that with Zoom's result:

![Zoom Background](https://raw.githubusercontent.com/floft/gstreamer-background-replace/master/files/zoom_screenshot.png)
