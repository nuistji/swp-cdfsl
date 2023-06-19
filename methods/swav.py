import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from .baselinetrain import BaselineTrain




class SwAV(nn.Module): # wrapper for BaselineTrain
    def __init__(self, base_encoder, normalize=True, output_dim=128, hidden_mlp=2048, nmb_prototypes=3000):    # based on the default values of SwAV code
        super(SwAV, self).__init__()
        assert isinstance(base_encoder, BaselineTrain)
        self.base_encoder = base_encoder

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