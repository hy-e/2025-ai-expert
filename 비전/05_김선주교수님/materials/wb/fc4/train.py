import os, argparse
import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, USE_CONFIDENCE_WEIGHTED_POOLING
from auxiliary.utils import print_metrics, log_metrics
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.data.LSMIDataset import get_loader
from classes.fc4.ModelFC4 import ModelFC4
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

EPOCHS = 2000
BATCH_SIZE = 1
LEARNING_RATE = 0.0003

# Which of the 3 folds should be processed (either 0, 1 or 2)
FOLD_NUM = 0

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "fold_{}".format(FOLD_NUM), "model.pth")


def main(config):
    path_to_log = os.path.join("logs", "fold_{}_{}".format(str(FOLD_NUM), str(time.time())))
    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")

    model = ModelFC4()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)
    model.set_optimizer(LEARNING_RATE)

    # Dataset & DataLoader
    train_loader = get_loader(config, 'train')
    val_loader = get_loader(config, 'val')
    
    # training_set = ColorCheckerDataset(train=True, folds_num=FOLD_NUM)
    # training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=20, drop_last=True)
    # print("\n Training set size ... : {}".format(len(training_set)))

    # test_set = ColorCheckerDataset(train=False, folds_num=FOLD_NUM)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=20, drop_last=True)
    # print(" Test set size ....... : {}\n".format(len(test_set)))

    # path_to_vis = os.path.join(path_to_log, "test_vis")
    # if TEST_VIS_IMG:
    #     print("Test vis for monitored image {} will be saved at {}\n".format(TEST_VIS_IMG, path_to_vis))
    #     os.makedirs(path_to_vis)

    print("\n**************************************************************")
    print("\t\t\t Training FC4 - Fold {}".format(FOLD_NUM))
    print("**************************************************************\n")

    evaluator = Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(EPOCHS):

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (img, label, _) in enumerate(train_loader):
            model.reset_gradient()
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model.predict(img)
            loss = model.optimize(pred, label)
            train_loss.update(loss)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} ]".format(epoch, EPOCHS, i, loss))

        train_time = time.time() - start

        val_loss.reset()
        start = time.time()

        if epoch % 5 == 0:
            evaluator.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():
                for i, (img, label, file_name) in enumerate(val_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    pred = model.predict(img)
                    loss = model.get_angular_loss(pred, label).item()
                    val_loss.update(loss)
                    evaluator.add_error(loss)

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} ]".format(epoch, EPOCHS, i, loss))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
        print("********************************************************************\n")

        # if 0 < val_loss.avg < best_val_loss:
        #     best_val_loss = val_loss.avg
        #     best_metrics = evaluator.update_best_metrics()
        #     print("Saving new best model... \n")
        #     model.save(os.path.join(path_to_log, str(val_loss.avg) + "_model.pth"))
        model.save(os.path.join(path_to_log, str(val_loss.avg) + "_model.pth"))

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--lr', type=float, default=1e-4)

    # dataset & loader config
    parser.add_argument('--image_pool', type=str, nargs='+', default=['1'])
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