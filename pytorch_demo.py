#!/usr/bin/env python3
"""
Replace a webcam background via GStreamer

Neural net segmentation, find the person in the image, and then extract the
person and place on a different background image

Based on:
https://github.com/kairess/semantic-segmentation-pytorch/blob/master/main.ipynb
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import cv2


def create_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor']
    return model, labels


def create_colors():
    cmap = plt.cm.get_cmap('tab20c')
    colors = (cmap(np.arange(cmap.N)) * 255).astype(np.int)[:, :3].tolist()
    np.random.seed(2020)
    np.random.shuffle(colors)
    colors.insert(0, [0, 0, 0])  # background color must be black
    colors = np.array(colors, dtype=np.uint8)

    return colors


def show_colors(colors):
    palette_map = np.empty((10, 0, 3), dtype=np.uint8)
    legend = []

    for i in range(21):
        legend.append(mpatches.Patch(color=np.array(colors[i]) / 255., label='%d: %s' % (i, labels[i])))
        c = np.full((10, 10, 3), colors[i], dtype=np.uint8)
        palette_map = np.concatenate([palette_map, c], axis=1)

    plt.figure(figsize=(20, 2))
    plt.legend(handles=legend)
    plt.imshow(palette_map)
    plt.pause(0.1)


def segment(model, colors, img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)['out'][0]  # (21, height, width)

    output_predictions = output.argmax(0).byte().cpu().numpy()  # (height, width)

    r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0]))
    r.putpalette(colors)

    return r, output_predictions


def show_segment(model, colors, filename, show=False):
    img = np.array(Image.open(filename))
    fg_h, fg_w, _ = img.shape

    segment_map, pred = segment(model, colors, img)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(img)
        axes[1].imshow(segment_map)
        plt.pause(0.1)

    return img, segment_map, pred, fg_h, fg_w


def load_background(filename, fg_w, fg_h, show=False):
    background = np.array(Image.open(filename))
    bg_h, bg_w, _ = background.shape

    # fit to fg width
    background = cv2.resize(background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))
    bg_h, bg_w, _ = background.shape
    margin = (bg_h - fg_h) // 2

    if margin > 0:
        background = background[margin:-margin, :, :]
    else:
        background = cv2.copyMakeBorder(background, top=abs(margin),
            bottom=abs(margin), left=0, right=0, borderType=cv2.BORDER_REPLICATE)

    # final resize
    background = cv2.resize(background, dsize=(fg_w, fg_h))

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(background)
        plt.pause(0.1)

    return background


def replace_background(model, colors, foreground_filename, background_filename,
        show=False):
    # Load segmentation and background
    img, segment_map, pred, fg_h, fg_w = show_segment(model, colors,
        foreground_filename)
    background = load_background(background_filename, fg_w, fg_h)

    # Separate foreground and background
    # mask = (pred == 15).astype(float) * 255  # 15: person
    mask = (pred != 0).astype(float) * 255  # 0: background, so everything else
    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0).astype(float)

    alpha = alpha / 255.  # (height, width)
    alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2)  # (height, width, 3)

    foreground = cv2.multiply(alpha, img.astype(float))
    background = cv2.multiply(1. - alpha, background.astype(float))

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].imshow(mask)
        axes[1].imshow(foreground.astype(np.uint8))
        axes[2].imshow(background.astype(np.uint8))

    # Final result
    result = cv2.add(foreground, background).astype(np.uint8)
    # Image.fromarray(result).save('imgs/result.jpg')

    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(result)
        plt.pause(0.1)

    return result


def show_result(result):
    plt.figure(figsize=(12, 12))
    plt.imshow(result)
    plt.pause(0.1)


if __name__ == "__main__":
    model, labels = create_model()

    colors = create_colors()
    # show_colors(colors)

    bg = "screenshot_20200328_130012.png"

    def seg(fn):
        result = replace_background(model, colors, fn, bg)
        show_result(result)

    seg("test2/image-0000001.jpg")
    seg("test2/image-0000090.jpg")
    seg("test2/image-0000320.jpg")

    # Wait to exit so we can look at the plots
    input("waiting")
