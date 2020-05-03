#!/bin/bash
# Make sure driver is loaded
echo "Loading v4l2loopback module"
modprobe v4l2loopback &>/dev/null || \
    sudo modprobe v4l2loopback video_nr=7 exclusive_caps=1

input="/dev/video2"
output="/dev/video7"
background="background.jpg"

if [[ ! -e $background ]]; then
    echo "Background $background does not exist"
    exit 1
fi

echo "Starting GStreamer background replace"
./main.py --input_device="$input" --output_device="$output" \
    --output_width=854 --output_height=480 --framerate=30 --green_tint=0.04 \
    --color_temp=0.7 --vignette_size=0.4 --vignette_soft=0.6 \
    --background="$background" "$@"
