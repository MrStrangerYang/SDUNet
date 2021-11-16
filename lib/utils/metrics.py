from torch import nn
import pdb
import numpy as np


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)
        alpha = 0.7

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        return (1 - alpha) * bce_loss + alpha * (1 - dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    def __init__(self, input, target, test=False):
        self.input = input
        self.target = target
        if not test:
            self.pred = input[:].squeeze().cpu().detach().numpy() > 0.5
            self.truth = np.asarray(target[:].squeeze().cpu().detach().numpy(), dtype=np.bool)
            self.num_in_target = self.input.size(0)
        else:
            self.pred = input > 0.5
            self.truth = np.asarray(target, dtype=np.bool)
            self.num_in_target = 1

        self.tp = np.logical_and(self.pred, self.truth)
        self.tn = np.logical_and(~self.pred, ~self.truth)

    def jaccard_index(self):
        intersection = (self.input * self.target).long().sum().detach().cpu()[0]
        union = self.input.long().sum().detach().cpu()[0] + self.target.long().sum().detach().cpu()[0] - intersection

        if union == 0:
            return float('nan')
        else:
            return float(intersection) / float(max(union, 1))

    def dice_coeff(self):
        smooth = 1e-15
        pred = self.input.view(self.num_in_target, -1)
        truth = self.target.view(self.num_in_target, -1)
        intersection = (pred * truth).sum(1)
        loss = (2. * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)
        return loss.mean().item()

    def mean_iou(self):
        ious = []
        for i in range(self.num_in_target):
            # pdb.set_trace()
            pred_i = self.pred[i]
            truth_i = self.truth[i]
            union = np.sum(np.logical_or(pred_i, truth_i))
            intersection = np.sum(np.logical_and(pred_i, truth_i))
            iou = float(intersection) / (float(union)+0.001)
            ious.append(iou)
        return np.sum(ious) / float(self.num_in_target)

    def prec(self):
        precs = []
        for i in range(self.num_in_target):
            tp_i = np.sum(self.tp[i])
            pred_i = np.sum(self.pred[i])
            if pred_i == 0:
                precision_i = 0
            else:
                precision_i = float(tp_i) / float(pred_i)
            precs.append(precision_i)
        return np.sum(precs) / float(self.num_in_target)

    def recall(self):
        recalls = []
        for i in range(self.num_in_target):
            tp_i = np.sum(self.tp[i])
            truth_i = np.sum(self.truth[i])
            if truth_i == 0:
                recall_i = 0
            else:
                recall_i = float(tp_i) / float(truth_i)
            recalls.append(recall_i)
        return np.sum(recalls) / float(self.num_in_target)

    def acc(self):
        accs = []
        for i in range(self.num_in_target):
            tp_i = np.sum(self.tp[i])
            tn_i = np.sum(self.tn[i])
            acc_i = float(tp_i + tn_i) / float(len(self.pred[i]))
            accs.append(acc_i)
        return np.sum(accs) / float(self.num_in_target)

    def f1_score(self):
        f1_scores = []
        for i in range(self.num_in_target):
            # pdb.set_trace()
            tp_i = np.sum(self.tp[i])
            pred_i = np.sum(self.pred[i])
            truth_i = np.sum(self.truth[i])
            # print(truth_i)
            recall_i = float(tp_i)+0.001 / (float(truth_i)+0.001)
            if pred_i == 0:
                f1_score_i = 0
            else:
                precision_i = float(tp_i) / float(pred_i) + 1e-3
                f1_score_i = 2 * precision_i * recall_i / (precision_i + recall_i)
            f1_scores.append(f1_score_i)
        return np.sum(f1_scores) / float(self.num_in_target)
