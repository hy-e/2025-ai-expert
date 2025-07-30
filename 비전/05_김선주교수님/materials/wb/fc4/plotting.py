import os, argparse, json

import torch.utils.data

import matplotlib.pyplot as plt

import numpy as np
from auxiliary.settings import DEVICE
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4
from classes.data.LSMIDataset import get_loader
from classes.training.Evaluator import Evaluator
from classes.data.LSMIDataset import get_loader

"""
* FC4 using confidence-weighted pooling (fc_cwp):

Fold	Mean		Median		Trimean		Best 25%	Worst 25%	Worst 5%
0	    1.73		1.47		1.50		0.50		3.53		4.20
1	    2.11		1.54		1.66		0.43		4.87		5.89
2	    1.92		1.45		1.52		0.52		4.22		5.66
Avg	    1.92		1.49		1.56		0.48		4.21		5.25
StdDev	0.19		0.05		0.09		0.05		0.67		0.92

* FC4 using summation pooling (fc_sum):

Fold	Mean		Median		Trimean		Best 25%	Worst 25%	Worst 5%	
0	    1.68        1.20	    1.35    	0.40	    3.71	    4.25
1	    2.11	    1.62	    1.68	    0.51	    4.74	    5.78
2	    1.79	    1.24	    1.35	    0.38	    4.21	    5.60
Avg	    1.86	    1.35	    1.46	    0.43	    4.22	    5.21
StdDev  0.22	    0.23	    0.19	    0.07	    0.52	    0.84
"""

MODEL_TYPE = "fc4_sum"

def pixel_level_angular_error(img1, img2): # [H, W, 3]
    img1_norm_per_pixel = np.linalg.norm(img1, axis=-1)[:, :, None] # [H, W, 1]
    img2_norm_per_pixel = np.linalg.norm(img2, axis=-1)[:, :, None] # [H, W, 1]
    
    img1_normalized = img1 / (img1_norm_per_pixel + 1e-6) # [H, W, 3]
    img2_normalized = img2 / (img2_norm_per_pixel + 1e-6) # [H, W, 3]

    print(img1[0][0])
    print(img1_normalized[0][0])
    print()
    print(img2[0][0])
    print(img2_normalized[0][0])
    print()
    print()
    
    return np.arccos(np.sum(img1_normalized * img2_normalized, axis=-1))

def main(config):
    evaluator = Evaluator()
    model = ModelFC4()
    num_fold = 0

    dataloader = get_loader(config, 'test')
    # test_set = ColorCheckerDataset(train=False, folds_num=num_fold)
    # dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)

    # Edit this path to point to the trained model to be tested
    path_to_pretrained = os.path.join("logs", "fold_0_1623484043.3203766", "1.2974605162938435_model.pth")
    model.load(path_to_pretrained)
    model.evaluation_mode()

    print("\n *** FOLD {} *** \n".format(num_fold))
    print(" * Using pretrained model stored at: {} \n".format(path_to_pretrained))
    
    with open('GALAXY_orgset/meta.json', 'r') as f:
        meta = json.load(f)

    with torch.no_grad():
        for i, (img, label, file_name) in enumerate(dataloader):
            img, label = img.to(DEVICE), label.to(DEVICE)

            # print(file_name)
            # pred_illum = model.predict(img)
            # pred = torch.ones_like(label[0]).detach().cpu().numpy()
            # # pred[:,:,0] = pred[:,:,0]*pred_illum[None, None, 0] # red
            # # pred[:,:,1] = pred[:,:,1]*pred_illum[None, None, 1] # red
            # # pred[:,:,2] = pred[:,:,2]*pred_illum[None, None, 2] # red
            # # print(pred)
            # # print(pred_illum)

            # pred = pred * pred_illum[None, None, :].detach().cpu().numpy() # red
            # # print(label[0].shape)
            # # print(pred_illum)
            # # print(pred[0])
            # # print(label[0])
            # # print(file_name)
            
            # error_map = pixel_level_angular_error(pred[0], label[0].detach().cpu().numpy())
            # # print(error_map)
            # loss = np.mean(error_map)
            # evaluator.add_error(loss.item())
            # print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name[0], i, loss.item()))
            pred = model.predict(img)
            sum = 0
            for i in range(label[0].shape[0]):
                for j in range(label[0].shape[1]):
                    error = model.get_angular_loss(pred, label[:, i, j])
                    # print(error)
                    sum += error
            loss = sum / (256*256)
            evaluator.add_error(loss.item())

            place_name = file_name[0].split('_')[0]
            illum_num = file_name[0].split('_')[1] + '_0'
            gt_illum_list = meta['test'][place_name][illum_num]

            for gt_illum in gt_illum_list:
                u = gt_illum[0]
                v = gt_illum[2]
                plt.scatter(u, v, c='k', label='gt')
            
            pred_cpu = pred.cpu()
            r = pred_cpu[0][0]
            g = pred_cpu[0][1]
            b = pred_cpu[0][2]

            u = r/g
            v = b/g

            plt.scatter(u, v, c='r', label='pred')
            plt.legend()
            plt.savefig('plot_pred/'+file_name[0]+'_'+str(loss.item())+'.png')
            plt.clf()
            
            # print(label[:,0,0])
            # loss = model.get_angular_loss(pred, label)
            # evaluator.add_error(loss.item())
            print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name[0], i, loss.item()))

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
    parser.add_argument('--image_pool', type=str, nargs='+', default=['12'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--input_type', type=str, default='rgb', choices=['rgb','uvl'])
    parser.add_argument('--output_type', type=str, default=None, choices=['illumination','uv','mixmap'])
    parser.add_argument('--mask_black', type=str, default=None)
    parser.add_argument('--mask_highlight', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=10)

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