import argparse
import sys

import coremltools as ct
import cv2
import numpy as np

from backend.utils import load_weight
from model.rfb_320 import create_rfb_net
from model.slim_320 import create_slim_net

parser = argparse.ArgumentParser(
    description='convert model')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
args = parser.parse_args()


def main():
    input_shape = (240, 320)  # H,W
    base_channel = 8 * 2
    num_classes = 2

    if args.net_type == 'slim':
        torch_path = "pytorch_pretrained/version-slim-320.pth"
        mapping_table = "mapping_tables/slim_320.json"
        model = create_slim_net(input_shape, base_channel, num_classes, post_processing=False)
    elif args.net_type == 'RFB':
        torch_path = "pytorch_pretrained/version-RFB-320.pth"
        mapping_table = "mapping_tables/rfb_320.json"
        model = create_rfb_net(input_shape, base_channel, num_classes, post_processing=False)
    else:
        print("The net type is wrong!")
        sys.exit(1)

    load_weight(model, torch_path, mapping_table)

    mlmodel = ct.convert(model)

    img = cv2.imread('imgs/test_input.jpg')
    h, w, _ = img.shape
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0
    img_resize = np.expand_dims(img_resize, axis=0)

    result = mlmodel.predict({"input_1": img_resize})

    print(result)


if __name__ == '__main__':
    main()
