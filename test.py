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
from torchsummary import summary
import numpy as np
from lib.utils.metrics import *
from lib.utils.config import read_run_cfg
from lib.datasets.mass import MassRoadBuilding
from lib.utils.show_plot_img import MultiImgPloter
from lib.utils.transforms import *
from torchvision.utils import make_grid,save_image
import pdb
# import wandb
import PIL.Image as Image
import warnings
import random

warnings.simplefilter("ignore", (UserWarning, FutureWarning, Warning))


def main(model_name, dataset, cache_file):
    cfg_run = read_run_cfg(model_name, dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg_run["gpus"]
    if cfg_run["model_name"] == 'UNet':
        from lib.models import unet
        model = unet.UNet()
        from lib.models import SUNet
        model = SUNet.SUNet()
    elif cfg_run["model_name"] == 'SDUNet':
        from lib.models import SDUNet
        model = SDUNet.defineSDUNet(n_classes=1)
    else:
        model = unet.UNet()
    model = nn.DataParallel(model).cuda()
    # print(cfg_run['test_model_state_dict'])
    checkpoint = torch.load(os.path.join(cfg_run['test_model_state_dict'], cache_file))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])
    mass_dataset_val = MassRoadBuilding('test512', 'test512', cfg_run, transforms=val_transform)
    val_dataloader = DataLoader(mass_dataset_val, batch_size=cfg_run["batch_size_test"], num_workers=2, shuffle=False)
    criterion = BCEDiceLoss()
    validation(val_dataloader, model, criterion, cfg_run)


def validation(valid_loader, model, criterion, cfg_run):
    # example_images = []
    valid_acc = MetricTracker()
    valid_loss = MetricTracker()

    # switch to evaluate mode
    model.eval()
    # with open("log.txt", 'a+') as f:
    #     f.write(''.format(i[0], i[1]))
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def viz(module, input, output):
        # logger.info(module)
        # img = make_grid(output[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=20, normalize=True, pad_value=1)
        # logger.info(img.shape)
        save_img_dir = cfg_run["result_dir_test"]
        save_output_path = os.path.join(save_img_dir,
                                        '{}_feature.png'.format( random.randint(1,1000)))
        save_image(output[0].detach().cpu().unsqueeze(dim=1),save_output_path, nrow=10, padding=20, normalize=True, pad_value=1)

    val_precision = MetricTracker()
    val_iou = MetricTracker()
    val_recall = MetricTracker()
    val_f1_score = MetricTracker()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="test {}".format(cfg_run["model_name"],ncols=50))):
        # get the inputs and wrap in Variable

        with torch.no_grad():
            inputs = data['img'].cuda()
            labels = data['mask'].cuda()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()

        outputs = model(inputs)
        # for name, m in model.named_modules():
        #     pdb.set_trace()
        outputs = torch.sigmoid(outputs)

        loss = criterion(outputs, labels)
        metrics = Metrics(outputs, labels)

        val_precision.update(metrics.prec(), outputs.size(0))
        val_recall.update(metrics.recall(), outputs.size(0))
        val_f1_score.update(metrics.f1_score(), outputs.size(0))
        val_iou.update(metrics.mean_iou(), outputs.size(0))

        valid_acc.update(metrics.dice_coeff(), outputs.size(0))
        valid_loss.update(loss.item(), outputs.size(0))
        # save_test_result(data, outputs, idx, cfg_run)
        # example_images.append(wandb.Image(Image.new('RGB', (100, 100), (25, 40, 80))))
    # wandb.log({"examples": example_images, "Test Accuracy": valid_acc.avg, "Test Loss": valid_loss.avg})
    tqdm.write(
        '{}| Validation Loss: {:.4f} | Acc: {:.4f} | precision {:.4f} | recall {:.4f} '
        '| iou {:.4f} | f1 score {:.4f} |\n'.format(cfg_run["model_name"],
                                                    valid_loss.avg, valid_acc.avg, val_precision.avg, val_recall.avg,
                                                    val_iou.avg, val_f1_score.avg))
    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg}


def save_test_result(data, outputs, idx, cfg_run):
    # options_run = read_run_cfg(model_name)
    examples = []
    # print('test')
    for i in range(data['img'].size()[0]):
        img_show = data['ori_img'][i, :, :, :].cpu().numpy()
        mask_show = data['mask'][i, :, :].squeeze().cpu().numpy()
        output_show = outputs[i, 0, :, :].detach().cpu().numpy()
        output_show[output_show >= 0.7] = 1
        output_show[output_show < 0.3] = 0
        # output_show = np.ones((500, 500), dtype=np.uint8)
        # output img & gt_mask
        ori_img_dir = os.path.join(cfg_run["result_dir_test"], 'img')
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
        save_img_dir = cfg_run["result_dir_test"]
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


if __name__ == '__main__':
    cache_file = 'check_point_149.pth.tar'
    # main('UNet','mass', cache_file)
    # main('SUNet','mass', cache_file)
    main('SDUNet','mass', cache_file)