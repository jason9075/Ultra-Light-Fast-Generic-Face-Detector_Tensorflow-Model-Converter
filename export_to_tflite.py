import platform
import time
import argparse

import cv2
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="export_models/slim/", type=str,
                    help='Folder of exported save model')
parser.add_argument('--out', default="export_models/model.tflite", type=str)
parser.add_argument('--edge_tpu', action="store_true")
args = parser.parse_args()

args.edge_tpu = False
EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def preprocess_image(img):
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0
    img_resize = np.float32(np.expand_dims(img_resize, axis=0))

    return img_resize


def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(args.save_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    open(args.out, "wb").write(tflite_model)


def test():
    if args.edge_tpu:
        interpreter = tflite.Interpreter(model_path=args.out,
                                         experimental_delegates=[
                                             tflite.load_delegate(EDGETPU_SHARED_LIB)])
    else:
        interpreter = tf.lite.Interpreter(model_path=args.out)

    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread('imgs/test_input.jpg')
    img_resize = preprocess_image(img)

    interpreter.set_tensor(input_details[0]['index'], img_resize)

    # first 3 times is warmup
    for _ in range(3):
        interpreter.invoke()
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    print(results)


if __name__ == '__main__':
    main()
    test()
