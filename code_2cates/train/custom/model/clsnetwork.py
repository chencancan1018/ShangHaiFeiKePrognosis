import numpy as np
import random
import timm
import torch
import torch.nn as nn
from starship.umtf.common.model import NETWORKS

p_mixup = 0.5

def mixup(input, truth, clip=[0,1]):
    indices = torch.randperm(input.size(0))
    shuffle_input = input[indices]
    shuffle_labels = truth[indices]
    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffle_input * (1 - lam)
    return input, truth, shuffle_labels, lam

def criterion(logits, targets):
    logits = logits.view(-1)
    targets = targets.view(-1)

    bce_loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = bce_loss_func(logits, targets)
    loss_sum = (2 * loss * (targets > 0)).sum() + (loss * (targets == 0)).sum()
    norm = (2 * (targets > 0)).sum() + (1 * (targets == 0)).sum() + 1
    return loss_sum / norm

loss_func = nn.CrossEntropyLoss()

@NETWORKS.register_module
class ClsNetwork25D(nn.Module):

    def __init__(self, backbone,
                in_ch=3,
                num_classes=3,
                drop_rate=0.,
                drop_rate_last=0.3,
                drop_path_rate=0.,
                loss_weight = (1, 7),
                pretrained=False,
                patch_size=(48, 256, 256),
                apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(ClsNetwork25D, self).__init__()

        self.in_ch = in_ch
        self._patch_size = patch_size
        self._loss_weight = loss_weight

        self.encoder = timm.create_model(
            backbone,
            in_chans=self.in_ch,
            num_classes=num_classes,
            features_only=False,
            drop_rate=0,
            pretrained=pretrained,
        )
        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        elif ('resnet' in backbone) or ('inception_v3' in backbone):
            hdim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )
        self.lstm2 = nn.LSTM(hdim, 256, num_layers=2, dropout=drop_rate, bidirectional=True, batch_first=True)
        self.head2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    def forward(self, vol, label):
        do_mixup = False
        images = vol
        labels = label
        # if random.random() < p_mixup:
        #     do_mixup = True
        #     images, labels, shuffle_labels, lam = mixup(images, labels)
            
        bs = images.shape[0]
        preds1, preds2 = self.forward_test(images)
        targets1 = labels.repeat(1, self._patch_size[0]).view(bs*self._patch_size[0], 1).squeeze()
        targets2 = labels.squeeze()

        # loss1 = criterion(preds1, targets1)
        # loss2 = criterion(preds2, targets2)
        loss1 = loss_func(preds1, targets1)
        loss2 = loss_func(preds2, targets2)
        # lw = [v / sum(self._loss_weight) for v in self._loss_weight]
        lw = self._loss_weight
        # if do_mixup:
        #     targets_mix1 = shuffle_labels.repeat(1, self._patch_size[0]).view(bs*self._patch_size[0], 1).squeeze()
        #     targets_mix2 = shuffle_labels.squeeze()
        #     loss11 = loss_func(preds1, targets_mix1)
        #     loss22 = loss_func(preds2, targets_mix2)
        #     loss1 = loss1  * lam  + loss11 * (1 - lam)
        #     loss2 = loss2 * lam  + loss22 * (1 - lam)
        loss1 = loss1 * lw[0]
        loss2 = loss2 * lw[1]
        return {"loss1": loss1, "loss2": loss2}

    def forward_test(self, images):
        bs = images.shape[0]
        images = images.view(bs*self._patch_size[0], self.in_ch, self._patch_size[1], self._patch_size[2])
        feat = self.encoder(images)
        feat = feat.view(bs, self._patch_size[0], -1)
        feat1, _ = self.lstm(feat)
        feat1 = feat1.contiguous().view(bs*self._patch_size[0], 512)
        preds1 = self.head(feat1)
        feat2, _ = self.lstm2(feat)
        preds2 = self.head2(feat2[:, 0])
        return preds1, preds2

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
        self.head2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.head2)

loss_func_none = nn.CrossEntropyLoss(reduction='none')

@NETWORKS.register_module
class ClsNetwork2D(nn.Module):

    def __init__(self, backbone,
                in_ch=3,
                num_classes=3,
                drop_rate=0.,
                drop_rate_last=0.3,
                drop_path_rate=0.,
                pretrained=False,
                patch_size=(256, 256),
                apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(ClsNetwork2D, self).__init__()

        self.in_ch = in_ch
        self._patch_size = patch_size

        self.encoder = timm.create_model(
            backbone,
            in_chans=self.in_ch,
            num_classes=num_classes,
            features_only=False,
            drop_rate=0,
            pretrained=pretrained,
        )
        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        elif ('resnet' in backbone) or ('inception_v3' in backbone):
            hdim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    def forward(self, vol, label):
        images = vol
        labels = label
        bs = images.shape[0]
        preds = self.forward_test(images)
        targets = labels.squeeze()

        loss = loss_func_none(preds, targets)
        # online hard examples mining
        # loss = loss_func_none(preds, targets)
        # ohem_loss, ohem_inds = torch.topk(loss, k=int(bs/4))
        # keep_num = bs + int(bs/4)
        # loss = (loss.sum() + ohem_loss.sum()) / keep_num
        
        classes = torch.unique(targets)
        values, _ = torch.sort(loss[targets==classes[0]])
        num = random.randint(20, 50)
        total_loss = values[:num].mean()
        for i in range(1, len(classes)):
            values, _ = torch.sort(loss[targets==classes[i]])
            total_loss = total_loss + values[:int(len(values) / 6)].mean()
        return {"loss": total_loss}

    def forward_test(self, images):
        feat = self.encoder(images)
        preds = self.head(feat)
        return preds

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)


@NETWORKS.register_module
class ClsNetwork2DSigHead(nn.Module):

    def __init__(self, backbone,
                in_ch=3,
                num_classes=1,
                drop_rate=0.,
                drop_rate_last=0.3,
                drop_path_rate=0.,
                pretrained=False,
                patch_size=(256, 256),
                apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(ClsNetwork2DSigHead, self).__init__()

        self.in_ch = in_ch
        self._patch_size = patch_size
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.encoder = timm.create_model(
            backbone,
            in_chans=self.in_ch,
            num_classes=num_classes,
            features_only=False,
            drop_rate=0,
            pretrained=pretrained,
        )
        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        elif ('resnet' in backbone) or ('inception_v3' in backbone):
            hdim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    def forward(self, vol, label):
        images = vol
        labels = label
        bs = images.shape[0]
        preds = self.forward_test(images)
        targets = labels.squeeze()
        preds = preds.view(-1)
        targets = targets.view(-1).float()

        loss = self.bce_loss_func(preds, targets)
        # online hard examples mining
        # loss = loss_func_none(preds, targets)
        # ohem_loss, ohem_inds = torch.topk(loss, k=int(bs/4))
        # keep_num = bs + int(bs/4)
        # loss = (loss.sum() + ohem_loss.sum()) / keep_num
        
        # # online multiple-instance learning
        # sort_values, sort_indexes = torch.sort(loss)
        # sort_targets = targets[sort_indexes]
        # num = random.randint(500, 600)
        # num = int(num / 2)
        # pos_loss = sort_values[sort_targets >= 0.5][:num]
        # neg_loss = sort_values[sort_targets < 0.5][:num]
        pos_loss = loss[targets >= 0.5]
        neg_loss = loss[targets < 0.5]
        pos_loss = pos_loss.sum() / (pos_loss.shape[0] + 1)
        neg_loss = neg_loss.sum() / (neg_loss.shape[0] + 1)
        total_loss = pos_loss + neg_loss
        return {"loss": total_loss}

    def forward_test(self, images):
        feat = self.encoder(images)
        preds = self.head(feat)
        return preds

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)