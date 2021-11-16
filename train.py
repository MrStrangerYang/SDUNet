from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm
import torch
import torch.optim as optim
import time
import shutil
import os
import cv2
from torch.optim.lr_scheduler import StepLR
import numpy as np
from lib.utils.metrics import *
from lib.utils.config import read_run_cfg
from lib.datasets.mass import MassRoadBuilding
from lib.datasets.deepGlobe import DGRoad
from lib.utils.show_plot_img import MultiImgPloter
from lib.utils.transforms import *
import pdb
# import wandb
import PIL.Image as Image
import warnings
from ptflops import get_model_complexity_info
import argparse

warnings.simplefilter("ignore", (UserWarning, FutureWarning, Warning))


def main(model_name, dataset):
    cfg_run = read_run_cfg(model_name, dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg_run["gpus"]
    print('-' * 8 + cfg_run["model_name"] + '-' * 8)

    if cfg_run["model_name"] == 'UNet':
        from lib.models import unet
        model = unet.UNet()
    elif cfg_run["model_name"] == 'SUNet':
        from lib.models import SUNet
        model = SUNet.SUNet()
    elif cfg_run["model_name"] == 'SDUNet':
        from lib.models import SDUNet
        model = SDUNet.defineSDUNet(n_classes=1)
    else:
        model = unet.UNet()
    # wandb.watch(model, log="all")
    model = nn.DataParallel(model).cuda()

    # set up binary cross entropy and dice loss
    criterion = BCEDiceLoss()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=cfg_run["learning_rate"], weight_decay=1e-5)

    # decay LR
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg_run["num_epoch_lr_decay"],
                                             gamma=cfg_run["lr_delay"])

    # starting params
    best_loss = 999

    # optionally resume from a checkpoint
    # if cfg.TRAIN.RESUME:
    #     if os.path.isfile(cfg.TRAIN.RESUME_PATH):
    #         print("=> loading checkpoint '{}'".format(cfg.TRAIN.RESUME_PATH))
    #         checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    #
    #         if checkpoint['epoch'] > cfg.TRAIN.START_EPOCH:
    #             START_EPOCH = checkpoint['epoch']
    #
    #         best_loss = checkpoint['best_loss']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAIN.RESUME_PATH, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(cfg.TRAIN.RESUME_PATH))

    # get data
    # pdb.set_trace()
    train_transform = DualCompose([
        RandomCrop((512, 512)),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ImageOnly(RandomHueSaturationValue()),
        ImageOnly(Normalize())
    ])
    val_transform = DualCompose([
        CenterCrop((512, 512)),
        ImageOnly(Normalize())
    ])
    if dataset == 'mass':
        mass_dataset_train = MassRoadBuilding('train', 'train', cfg_run, transforms=train_transform)
        # mass_dataset_val = MassRoadBuilding('valid', transforms=val_transform)
        mass_dataset_val = MassRoadBuilding('valid', 'valid', cfg_run, transforms=val_transform)
    elif dataset == 'deepglobe':
        mass_dataset_train = DGRoad('train', 'train_crops', cfg_run, transforms=train_transform)
        # mass_dataset_val = MassRoadBuilding('valid', transforms=val_transform)
        mass_dataset_val = DGRoad('valid', 'test_all_rgb-ps', cfg_run, transforms=val_transform)

    train_dataloader = DataLoader(mass_dataset_train, batch_size=cfg_run["batch_size_train"], num_workers=2,
                                  shuffle=True)
    # val_dataloader = DataLoader(mass_dataset_val, batch_size=cfg_run["batch_size_test"], num_workers=2, shuffle=False)

    for epoch in range(cfg_run["start_epoch"], cfg_run["epoch_train"]):
        print('Epoch {}/{}'.format(epoch, cfg_run["epoch_train"] - 1))
        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        train(train_dataloader, model, criterion, optimizer, lr_scheduler, cfg_run)
        # valid_metrics = validation(val_dataloader, model, criterion, epoch, cfg_run)

        # is_best = valid_metrics['valid_loss'] < best_loss
        # best_loss = min(valid_metrics['valid_loss'], best_loss)
        # save_checkpoint(state, is_best, epoch, cfg_run):
        save_checkpoint({
            'epoch': epoch,
            'arch': cfg_run["model_name"],
            'state_dict': model.state_dict(),
            'best_loss': '*',
            'optimizer': optimizer.state_dict()
        }, False, epoch, cfg_run)


def save_checkpoint(state, is_best, epoch, cfg_run):
    # cfg_run = read_run_cfg(model_name)
    # print(cfg_run["model_cache"])
    if not os.path.exists(cfg_run["model_cache"]):
        os.makedirs(cfg_run["model_cache"])
    if epoch % 20 == 19:
        filename = os.path.join(cfg_run["model_cache"], 'check_point_{0:3d}.pth.tar'.format(epoch))
        torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename,os.path.join(cfg.MODEL_CACHE,'model_best.pth.tar'))
        filename = os.path.join(cfg_run["model_cache"], 'model_best.pth.tar'.format(epoch))
        torch.save(state, filename)

def train(train_loader, model, criterion, optimizer, scheduler, cfg_run):
    model.train()
    train_acc = MetricTracker()
    train_loss = MetricTracker()

    scheduler.step()

    # iterate over data
    for idx, data in enumerate(tqdm(train_loader, desc="training")):
        # print(idx)
        # get the inputs and wrap in Variable
        inputs = data['img'].cuda()
        labels = data['mask'].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        optimizer.step()

        train_acc.update(Metrics(outputs, labels).dice_coeff(), outputs.size(0))
        train_loss.update(loss.item(), outputs.size(0))

    tqdm.write('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss.avg, train_acc.avg))

    return {'train_loss': train_loss.avg, 'train_acc': train_acc.avg}


def validation(valid_loader, model, criterion, epoch, cfg_run):
    example_images = []
    valid_acc = MetricTracker()
    valid_loss = MetricTracker()

    # switch to evaluate mode
    model.eval()

    val_precision = MetricTracker()
    val_iou = MetricTracker()
    val_recall = MetricTracker()
    val_f1_score = MetricTracker()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="testing")):
        # get the inputs and wrap in Variable

        with torch.no_grad():
            inputs = data['img'].cuda()
            labels = data['mask'].cuda()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()

        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)

        loss = criterion(outputs, labels)
        metrics = Metrics(outputs, labels)

        val_precision.update(metrics.prec(), outputs.size(0))
        val_recall.update(metrics.recall(), outputs.size(0))
        val_f1_score.update(metrics.f1_score(), outputs.size(0))
        val_iou.update(metrics.mean_iou(), outputs.size(0))

        valid_acc.update(metrics.dice_coeff(), outputs.size(0))
        valid_loss.update(loss.item(), outputs.size(0))
        if epoch % 20 == 19:
            save_val_result(data, outputs, epoch, idx, cfg_run)
        # example_images.append(wandb.Image(Image.new('RGB', (100, 100), (25, 40, 80))))

    # wandb.log({"examples": example_images, "Test Accuracy": valid_acc.avg, "Test Loss": valid_loss.avg})

    tqdm.write(
        '| Validation Loss: {:.4f} | Acc: {:.4f} | precision {:.4f} | recall {:.4f} '
        '| iou {:.4f} | f1 score {:.4f} |'.format(
            valid_loss.avg, valid_acc.avg, val_precision.avg, val_recall.avg, val_iou.avg, val_f1_score.avg))
    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg}


# create a function to save the model state (https://github.com/pytorch/examples/blob/master/imagenet/main.py)



def save_val_result(data, outputs, epoch_num, idx, cfg_run):
    # options_run = read_run_cfg(model_name)
    examples = []
    for i in range(data['img'].size()[0]):
        img_show = data['ori_img'][i, :, :, :].cpu().numpy()
        mask_show = data['mask'][i, :, :].squeeze().cpu().numpy()
        output_show = outputs[i, 0, :, :].detach().cpu().numpy()
        output_show[output_show >= 0.5] = 1
        output_show[output_show < 0.5] = 0
        # output_show = np.ones((500, 500), dtype=np.uint8)
        # output img & gt_mask
        if epoch_num == 0:
            ori_img_dir = os.path.join(cfg_run["result_dir"], 'img')
            if not os.path.exists(ori_img_dir):
                os.makedirs(ori_img_dir)
            save_ori_path = os.path.join(ori_img_dir,
                                         '{}_ori.jpg'.format(idx * cfg_run["batch_size_test"] + i))
            save_mask_path = os.path.join(ori_img_dir,
                                          '{}_mask.jpg'.format(idx * cfg_run["batch_size_test"] + i))
            cv2.imwrite(os.path.join(save_ori_path), img_show)
            cv2.imwrite(os.path.join(save_mask_path), mask_show * 255)

        img_to_show = MultiImgPloter(img1=img_show, img2=mask_show, img3=output_show).get_mplimage()
        examples.append(img_show)
        save_img_dir = os.path.join(cfg_run["result_dir"], 'epoch{0:03d}'.format(epoch_num))
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        save_img_path = os.path.join(save_img_dir, '{}.png'.format(idx * cfg_run["batch_size_test"] + i))
        cv2.imwrite(os.path.join(save_img_path), img_to_show)
        save_output_path = os.path.join(save_img_dir,
                                        '{}_output.png'.format(idx * cfg_run["batch_size_test"] + i))
        cv2.imwrite(os.path.join(save_output_path), output_show * 255)
        # np.save(os.path.join(save_img_dir, 'img {0:3d}.npy'.format(idx * cfg.VAL.BATCH_SIZE + i)), img_show)

        # np.save(os.path.join(save_img_dir, 'mask {0:3d}.npy'.format(idx * cfg.VAL.BATCH_SIZE + i)), mask_show)
        # np.save(os.path.join(save_img_dir, 'output {0:3d}.npy'.format(idx * cfg.VAL.BATCH_SIZE + i)), output_show)
    return examples


def get_model_complexity(model_name, dataset):
    cfg_run = read_run_cfg(model_name, dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg_run["gpus"]
    print('-' * 8 + cfg_run["model_name"] + '-' * 8)

    if cfg_run["model_name"] == 'UNet':
        from lib.models import unet
        model = unet.UNet()
    elif cfg_run["model_name"] == 'SUNet':
        from lib.models import SUNet
        model = SUNet.SUNet()
    elif cfg_run["model_name"] == 'SDUNet':
        from lib.models import SDUNet
        model = SDUNet.defineSDUNet(n_classes=1)
    else:
        model = unet.UNet()

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    # on vs code

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True
    )
    args = parser.parse_args()

    try:
        main(args.model_name, 'mass')
    except BaseException:
        print(args.model_name)
