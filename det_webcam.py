import argparse
import sys

import cv2
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(
    description='convert model')

parser.add_argument('--net_type', default="slim", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
args = parser.parse_args()


def main():
    if args.net_type == 'slim':
        model_path = "export_models/slim/"
    elif args.net_type == 'RFB':
        model_path = "export_models/RFB/"
    else:
        print("The net type is wrong!")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, origin_img = cap.read()

        h, w, _ = origin_img.shape
        img = cv2.resize(origin_img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img - 127.0
        img = img / 128.0

        results = model.predict(np.expand_dims(img, axis=0))

        for result in results:
            start_x = int(result[-4] * w)
            start_y = int(result[-3] * h)
            end_x = int(result[-2] * w)
            end_y = int(result[-1] * h)

            cv2.rectangle(origin_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 0)

        cv2.imshow('frame', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
