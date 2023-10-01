#!/usr/bin/env python

import argparse
import base64
import json
import os

import matplotlib.pyplot as plt

from labelme import utils

def main():
    parser = argparse.ArgumentParser()
    # Distionary of json file
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    # Load json file
    data = json.load(open(json_file))

    # Check if the image data is encoded in base64
    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    
    # Convert the base64 image data to a NumPy array
    img = utils.img_b64_to_arr(imageData)
    # (w, h) = img.size
    # print("w=%d, h=%d", w, h)
    # print(img.shape)

    # Process the label names and values
    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    
    # Convert shapes to a label array
    lbl, _  = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    # Visualize the image and labels
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    print(f"lbl: {len(lbl)}\n{lbl}")
    print(f"img: {img.shape}")
    print(f"label_names: {label_names}")
    lbl_viz = utils.draw_label(lbl, img, label_names)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl_viz)
    plt.show()


if __name__ == '__main__':
    main()