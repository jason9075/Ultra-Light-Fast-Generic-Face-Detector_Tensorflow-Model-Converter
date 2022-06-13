import numpy as np
import tensorflow as tf

from backend.op import conv_bn, conv_dw, basic_rfb, separable_conv
from backend.utils import concat_result, add_post_processing

conf_threshold = 0.6
nms_iou_threshold = 0.3
nms_max_output_size = 200
top_k = 100
center_variance = 0.1
size_variance = 0.2

image_size = [320, 240]  # default input size 320*240
feature_map_wh_list = [[40, 30], [20, 15], [10, 8], [5, 4]]  # default feature map size
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]


def create_rfb_net(input_shape, base_channel, num_classes, post_processing=True):
    input_node = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))

    net = conv_bn(input_node, base_channel, stride=2, prefix='basenet.0')  # 120x160
    net = conv_dw(net, base_channel * 2, stride=1, prefix='basenet.1')
    net = conv_dw(net, base_channel * 2, stride=2, prefix='basenet.2')  # 60x80
    net = conv_dw(net, base_channel * 2, stride=1, prefix='basenet.3')
    net = conv_dw(net, base_channel * 4, stride=2, prefix='basenet.4')  # 30x40
    net = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.5')
    net = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.6')
    header_0 = basic_rfb(net, base_channel * 4, base_channel * 4, stride=1, scale=1.0, prefix='basenet.7')
    net = conv_dw(header_0, base_channel * 8, stride=2, prefix='basenet.8')  # 15x20
    net = conv_dw(net, base_channel * 8, stride=1, prefix='basenet.9')
    header_1 = conv_dw(net, base_channel * 8, stride=1, prefix='basenet.10')
    net = conv_dw(header_1, base_channel * 16, stride=2, prefix='basenet.11')  # 8x10
    header_2 = conv_dw(net, base_channel * 16, stride=1, prefix='basenet.12')

    out = tf.keras.layers.Conv2D(base_channel * 4, 1, padding='SAME', name='extras_convbias')(header_2)
    out = tf.keras.layers.ReLU(name='extras_relu1')(out)
    out = separable_conv(out, base_channel * 16, kernel_size=3, stride=2, padding=1,
                         prefix='extras_sep')
    header_3 = tf.keras.layers.ReLU(name='extras_relu2')(out)

    reg_0 = separable_conv(header_0, 3 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_0_sep')
    cls_0 = separable_conv(header_0, 3 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_0_sep')

    reg_1 = separable_conv(header_1, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_1_sep')
    cls_1 = separable_conv(header_1, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_1_sep')

    reg_2 = separable_conv(header_2, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_2_sep')
    cls_2 = separable_conv(header_2, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_2_sep')

    reg_3 = tf.keras.layers.Conv2D(3 * 4, kernel_size=3, padding='SAME',
                                   name='reg_3_convbias')(header_3)
    cls_3 = tf.keras.layers.Conv2D(3 * num_classes, kernel_size=3, padding='SAME',
                                   name='cls_3_convbias')(header_3)

    reg_list = [reg_0, reg_1, reg_2, reg_3]
    cls_list = [cls_0, cls_1, cls_2, cls_3]

    def size(shape):
        return np.prod([a for a in shape if a is not None])

    reg_list = [tf.keras.layers.Reshape([int(size(reg.shape) / 4), 4])(reg) for reg in reg_list]
    cls_list = [tf.keras.layers.Reshape([int(size(cls.shape) / num_classes), num_classes])(cls) for cls in cls_list]

    reg = tf.keras.layers.Concatenate(axis=1)(reg_list)
    cls = tf.keras.layers.Concatenate(axis=1)(cls_list)

    cls = tf.keras.layers.Softmax(axis=-1)(cls)

    result = [cls, reg]

    if post_processing:
        result = concat_result([reg_0, reg_1, reg_2, reg_3],
                            [cls_0, cls_1, cls_2, cls_3],
                            num_classes, image_size, feature_map_wh_list, min_boxes,
                            center_variance, size_variance)

        result = add_post_processing(result)

    model = tf.keras.Model(inputs=[input_node], outputs=[result])
    model.summary()

    return model
