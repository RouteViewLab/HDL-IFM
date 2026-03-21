import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import SparseDatasetOnline, SparseDatasetOffline
import os
import torch.multiprocessing
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import cv2
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified)

from models.superglue import SuperGlue
from models.superglueLoss import superglueLoss
from dataset.data_builder import DataBuilder

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


def configParser():
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--keypoint_threshold', type=float,
        default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization based on OpenCV instead of Matplotlib')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001,
        help='Learning rate')
    parser.add_argument(
        '--eval_output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optional,'
             'visualizations are written')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.1,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=-1,
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--max_keypoints', type=int, default=800,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--feature_dim', type=int, default=256, help='superpoint feature dim')
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='batch_size')
    parser.add_argument('--train_path', type=str,
                        default=r"C:\Users\Administrator\PycharmProjects\pythonProject\datasets\COCO/",
                        help='Path to the directory of training imgs.')
    parser.add_argument('--hand_path', type=str,
                        default="",
                        help='Path to the directory of hand images')
    parser.add_argument('--superpoint_weight', type=str,
                        default="superpoint_v1.pth")
    parser.add_argument('--pretrained', type=str,
                        default=r"D:\checkpoint\superpoint_my_data\checkpoints\superglue_outdoor.pth")
    parser.add_argument('--debug', type=int, default=0)

    parser.add_argument('--dataset_online', type=int, default=0)
    parser.add_argument('--dataset_offline_rebuild', type=int, default=1)
    parser.add_argument(
        '--epoch', type=int, default=200,
        help='Number of epoches')
    parser.add_argument('--tensorboardLabel', type=str,
                        default='innerNeg1',
                        )

    opt = parser.parse_args()
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
            'feature_dim': opt.feature_dim,
            'weights_path': opt.superpoint_weight,  # feature_dim=128
        },
        'superglue': {
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'descriptor_dim': opt.feature_dim,
            'pretrained': opt.pretrained,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    return opt, config
if __name__ == '__main__':
    opt, config = configParser()
    print(opt)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    cudaKey = set(['keypoints0', 'keypoints1', 'descriptors0', 'descriptors1', 'scores0', 'scores1'])
    superglue = SuperGlue(config.get('superglue', {})).to(device)
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to',
          'directory \"{}\"'.format(eval_output_dir))
    def evaluate(model, loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                for k in cudaKey:
                    data[k] = data[k].to(device)
                pred = model(data)
                loss = superglueLoss(pred, data)
                # 确保 loss 是一个单一的值
                if isinstance(loss, tuple):
                    loss = loss[0]
                total_loss += loss.item()
        return total_loss / len(loader)
    if opt.dataset_online:
        dataBuilder = DataBuilder(config['superpoint'], r'C:\Users\Administrator\PycharmProjects\pythonProject\superglue_train/dataset/warped/', r'C:\Users\Administrator\PycharmProjects\pythonProject\superglue_train/dataset/sp/', numProcess=1,
                                  debug=opt.debug)
        train_set = SparseDatasetOnline(opt.train_path, opt.hand_path, dataBuilder)
    else:
        if opt.dataset_offline_rebuild:
            dataBuilder = DataBuilder(config['superpoint'], r'C:\Users\Administrator\PycharmProjects\pythonProject\superglue_train/dataset/warped/', r'C:\Users\Administrator\PycharmProjects\pythonProject\superglue_train/dataset/sp/', numProcess=1)
            dataBuilder.buildAll(opt.train_path, opt.hand_path, batchSizeMax=32, saveFlag=1, debug=opt.debug)
        train_set = SparseDatasetOffline(r'C:\Users\Administrator\PycharmProjects\pythonProject\superglue_train/dataset/sp/')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size,
                                             drop_last=True)

    superglue = SuperGlue(config.get('superglue', {})).to(device)
    if torch.cuda.is_available():
        superglue.cuda()
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    N = train_loader.dataset.__len__() // opt.batch_size

    writer =SummaryWriter("./logs/" + opt.tensorboardLabel)
    cudaKey = set(['keypoints0', 'keypoints1', 'descriptors0', 'descriptors1', 'scores0', 'scores1'])
    loss_values = []
    num_matches_values = []
    scores_values = []
    epoch_num_matches = []
    best_loss = float('inf')
    early_stopping_patience = 300
    no_improve_epochs = 0

    for epoch in range(0, opt.epoch):
        epoch_loss = 0
        epoch_num_matches.clear()
        superglue.train()
        for i, data in enumerate(train_loader):
            for k in cudaKey:
                data[k] = data[k].cuda()
            pred = superglue(data)
            Loss = superglueLoss(pred, data)
            loss_mean, num_matches, selected_scores = superglueLoss(pred, data)
            loss_values.append(loss_mean.item())
            num_matches_values.append(num_matches)
            scores_values.append(selected_scores.detach().cpu().numpy())
            epoch_num_matches.append(num_matches)

            epoch_loss += loss_mean.item()

            superglue.zero_grad()
            loss_mean.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{opt.epoch}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss_values[-1]:.4f}, '
                      f'Num Matches: {num_matches_values[-1]}, '
                      )

        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{opt.epoch}] done. Epoch Loss {epoch_loss:.4f}")
        epoch_avg_matches = np.mean(epoch_num_matches)
        print(f"Epoch [{epoch + 1}/{opt.epoch}] done. Epoch Loss: {epoch_loss:.4f}, "
              f"Average Num Matches: {epoch_avg_matches}")

        val_loss = evaluate(superglue, val_loader)
        print(f"Validation Loss: {val_loss}")

        if val_loss < best_loss:

            best_loss = val_loss
            no_improve_epochs = 0
            torch.save(superglue.state_dict(), "best_model.pth")
            print("New best model saved.")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")

            if no_improve_epochs >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        model_out_path = "E:\supergluetrainweight\model_epoch_{}.pth".format(epoch)
        torch.save(superglue.state_dict(), model_out_path)
        print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
              .format(epoch + 1, opt.epoch, epoch_loss, model_out_path))

    # plt.figure()
    # plt.plot(loss_values, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Over Epochs')
    # plt.legend()
    # plt.savefig('loss_epochs.png')
    # plt.figure()
    # plt.plot(epoch_num_matches, label='Avg num_matches')
    # plt.xlabel('Epoch')
    # plt.ylabel('Avg num_matches')
    # plt.title('Average Num Matches Over Epochs')
    # plt.legend()
    # plt.savefig('Average num_matchs_epochs.png')