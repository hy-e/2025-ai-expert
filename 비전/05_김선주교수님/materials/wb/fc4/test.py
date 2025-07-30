import os, argparse
import cv2
import torch.utils.data
from tqdm import tqdm

import numpy as np
from auxiliary.settings import DEVICE
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4
from classes.data.LSMIDataset import get_loader
from classes.training.Evaluator import Evaluator
from classes.data.LSMIDataset import get_loader

from visualize_ours import *
    

def main(config):
    evaluator = Evaluator()
    model = ModelFC4()
    num_fold = 0

    dataloader = get_loader(config, 'test')

    # Edit this path to point to the trained model to be tested
    path_to_pretrained = os.path.join('checkpoint', "pretrained_model.pth")
    model.load(path_to_pretrained)
    model.evaluation_mode()

    # print("\n *** FOLD {} *** \n".format(num_fold))
    print(" * Using pretrained model stored at: {} \n".format(path_to_pretrained))

    with torch.no_grad():
        for i, (img, label, file_name) in enumerate(tqdm(dataloader)):
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            pred = model.predict(img)
            sum = 0
            for i in range(label[0].shape[0]):
                for j in range(label[0].shape[1]):
                    error = model.get_angular_loss(pred, label[:, i, j])
                    sum += error
            loss = sum / (256*256)
            evaluator.add_error(loss.item())
            
            # visualize
            img_input = cv2.cvtColor(img[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
            pred_normalized = np.ones_like(pred)
            pred_normalized[:, 0] = pred[:, 0] / pred[:, 1] # normalize by G
            pred_normalized[:, 2] = pred[:, 2] / pred[:, 1] # normalize by G
            img_wb = img_input / pred_normalized
            img_wb = img_wb
            place_name = file_name[0].split('.')[0]
            img_gt = cv2.imread(os.path.join(config.data_root, 'test', place_name+"_gt.tiff"), cv2.IMREAD_UNCHANGED)

            vis_result = visualize(img_input, img_wb, img_gt, data_root='./') 
            cv2.imwrite(f'results/{place_name}.png', vis_result)

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--lr', type=float, default=1e-4)

    # dataset & loader config
    parser.add_argument('--image_pool', type=str, nargs='+', default=['1', '12'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--input_type', type=str, default='rgb', choices=['rgb','uvl'])
    parser.add_argument('--output_type', type=str, default=None, choices=['illumination','uv','mixmap'])
    parser.add_argument('--mask_black', type=str, default=None)
    parser.add_argument('--mask_highlight', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)

    # path config
    parser.add_argument('--data_root', type=str, default='GALAXY_orgset')
    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--result_root', type=str, default='results')
    parser.add_argument('--log_root', type=str, default='logs')
    parser.add_argument('--checkpoint', type=str, default='210520_0600')

    # Misc
    parser.add_argument('--save_epoch', type=int, default=-1,
                        help='number of epoch for auto saving, -1 for turn off')
    parser.add_argument('--multi_gpu', type=int, default=1, choices=[0,1],
                        help='0 for single-GPU, 1 for multi-GPU')

    config = parser.parse_args()
    main(config)