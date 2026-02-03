import argparse
import os

from config import cfg
from datasets.make_dataloader_promptsg import make_dataloader
from model.make_model_promptsg import make_model
from processor.processor_promptsg import do_inference, do_inference_with_visualization
from utils.logger import setup_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PromptSG Testing')
    parser.add_argument('--config_file', default='configs/person/vit_promptsg.yml', type=str)
    parser.add_argument('--visualize', action='store_true',
                        help='Generate attention map visualizations during inference')
    parser.add_argument('--num_visualize', type=int, default=20,
                        help='Number of images to visualize (default: 20)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('promptsg', output_dir, if_train=False)
    logger.info(args)

    if args.config_file != '':
        logger.info('Loaded configuration file {}'.format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info('\n' + cf.read())
    logger.info('Running with config:\n{}'.format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    if args.visualize:
        logger.info(f'Running inference with attention visualization for {args.num_visualize} images')
        do_inference_with_visualization(cfg, model, val_loader, num_query, output_dir, args.num_visualize)
    else:
        do_inference(cfg, model, val_loader, num_query)
