import argparse
import sys

from backend.utils import load_weight
from model.rfb_320 import create_rfb_net
from model.slim_320 import create_slim_net

parser = argparse.ArgumentParser(
    description='convert model')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--pytorch_model', default=None, type=str)
parser.add_argument('--postprocess', action="store_true")
args = parser.parse_args()


def main():
    input_shape = (240, 320)  # H,W
    base_channel = 8 * 2
    num_classes = 2

    if args.net_type == 'slim':
        torch_path = args.pytorch_model or "pytorch_pretrained/version-slim-320.pth"
        mapping_table = "mapping_tables/slim_320.json"
        model = create_slim_net(input_shape, base_channel, num_classes, post_processing=args.postprocess)
    elif args.net_type == 'RFB':
        torch_path = args.pytorch_model or "pytorch_pretrained/version-RFB-320.pth"
        mapping_table = "mapping_tables/rfb_320.json"
        model = create_rfb_net(input_shape, base_channel, num_classes, post_processing=args.postprocess)
    else:
        print("The net type is wrong!")
        sys.exit(1)

    load_weight(model, torch_path, mapping_table)
    model.save(f'export_models/{args.net_type}/', include_optimizer=False)


if __name__ == '__main__':
    main()
