import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from torchvision import transforms as T
from methods.baselinetrain import BaselineTrain


@torch.no_grad()
def distributed_sinkhorn(out, params):
    Q = torch.exp(out / 0.05).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    # for it in range(2):
    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class SwAV(nn.Module): # wrapper for BaselineTrain
    def __init__(self, base_encoder, normalize=True, output_dim=128, hidden_mlp=2048, nmb_prototypes=3000):    # based on the default values of SwAV code
        super(SwAV, self).__init__()
        assert isinstance(base_encoder, BaselineTrain)
        self.base_encoder = base_encoder
        self.final_feat_dim = 512

        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(base_encoder.feature.final_feat_dim, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(base_encoder.feature.final_feat_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

            # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            raise NotImplementedError('for MultiPrototypes')
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

    def forward_features(self, x, feature_selector: str = None):
        """
        You'll likely need to override this method for SSL models.
        """
        return self.base_encoder(x)

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = F.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        # not multi-crop setting for now...
        outputs = self.base_encoder.feature(torch.cat([inputs[0], inputs[1]], dim=0).cuda(non_blocking=True))
        return self.forward_head(outputs)

    def on_step_start(self):
        pass

    def on_step_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def compute_ssl_loss(self, X, return_features=False):
        X[0] = X[0].cuda()
        X[1] = X[1].cuda()
        nmb_crops = [2]  # for multi-crop, use eg. [2,6]
        size_crops = [224]  # for multi-crop, use eg. [224, 84]
        crops_for_assign = [0, 1]
        temperature = 0.1
        queue_length = 0
        freeze_prototypes_niters = 313
        sinkhorn_params = {'epsilon': 0.05,
                           'sinkhorn_iterations': 3}
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # iteration = epoch * len(unlabeled_target_loader) + i
        embedding, output = self(X)
        embedding = embedding.detach()
        bs = X[0].size(0)

        # ============ swav loss ... ============
        swav_loss = 0
        for j, crop_id in enumerate(crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                # if queue is not None:
                #     if not torch.all(queue[j, -1, :] == 0):
                #         out = torch.cat((torch.mm(
                #             queue[j],
                #             self.prototypes.weight.t()
                #         ), out))
                #     # fill the queue
                #     queue[j, bs:] = queue[j, :-bs].clone()
                #     queue[j, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out, sinkhorn_params)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):  # crop_id -> 0, v -> 1
                x = output[bs * v: bs * (v + 1)] / temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            swav_loss += subloss / (np.sum(nmb_crops) - 1)
        swav_loss /= len(crops_for_assign)
        return swav_loss

    def compute_cls_loss_and_accuracy(self, x, y, return_predictions=False) -> Tuple:
        features_base = self.base_encoder.feature(x.cuda())
        logits_base = self.base_encoder.classifier(features_base)
        log_probability_base = F.log_softmax(logits_base, dim=1)
        nll_criterion = nn.NLLLoss(reduction='mean').cuda()
        loss = nll_criterion(log_probability_base, y.cuda())
        return loss
        # scores = self.forward(x)
        # _, predicted = torch.max(scores.data, 1)
        # accuracy = predicted.eq(y.data).cpu().sum() / x.shape[0]
        # if return_predictions:
        #     return self.cls_loss_function(scores, y), accuracy, predicted
        # else:
        #     return self.cls_loss_function(scores, y), accuracy