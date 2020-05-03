#!/bin/bash
# Make sure driver is loaded
echo "Loading v4l2loopback module"
modprobe v4l2loopback &>/dev/null || sudo modprobe v4l2loopback video_nr=7 exclusive_caps=1

output=""
background=""

output="--output_device=/dev/video7"
background="--background=background.jpg"

echo "Starting GStreamer background replace"
./gstreamer_replace.py --input_device="/dev/video2" $output --output_width=854 --output_height=480 --framerate=30 --green_tint=0.04 --color_temp=0.7 --vignette_size=0.4 --vignette_soft=0.6 $background  "$@"