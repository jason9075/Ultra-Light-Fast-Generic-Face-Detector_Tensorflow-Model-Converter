import platform
import time

import cv2
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

SAVE_MODEL_DIR = 'export_models/slim/'
OUTPUT_TF_FILE_NAME = 'export_models/slim.tflite'

USE_EDGE_TPU = False
EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    open(OUTPUT_TF_FILE_NAME, "wb").write(tflite_model)


def test():
    if USE_EDGE_TPU:
        interpreter = tflite.Interpreter(model_path=OUTPUT_TF_FILE_NAME,
                                         experimental_delegates=[
                                             tflite.load_delegate(EDGETPU_SHARED_LIB)])
    else:
        interpreter = tf.lite.Interpreter(model_path=OUTPUT_TF_FILE_NAME)

    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread('imgs/test_input.jpg')
    h, w, _ = img.shape
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0

    img_resize = np.float32(np.expand_dims(img_resize, axis=0))

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
