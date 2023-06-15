import json
import os

import numpy as np
import pandas as pd
import torch
import torch.optim
from tqdm import tqdm
import backbone
import torch.nn as nn
from model.swav import SwAV
import torch.nn.functional as F
import time

from backbone import get_backbone_class
from datasets import ISIC_few_shot
from datasets.dataloader import get_dataloader, get_unlabeled_dataloader
from io_utils import parse_args
from model import get_model_class
from methods.baselinetrain import BaselineTrain
from paths import get_output_directory, get_final_pretrain_state_path, get_pretrain_state_path_pr, \
    get_pretrain_params_path, get_pretrain_history_path
from itertools import cycle


def _get_dataloaders(params):
    labeled_source_bs = params.ls_batch_size
    batch_size = params.batch_size
    unlabeled_source_bs = batch_size
    unlabeled_target_bs = batch_size

    if params.us and params.ut:
        unlabeled_source_bs //= 2
        unlabeled_target_bs //= 2

    ls, us, ut = None, None, None
    if params.ls:
        print('Using source data {} (labeled)'.format(params.source_dataset))
        ls = get_dataloader(dataset_name=params.source_dataset, augmentation=params.augmentation,
                            batch_size=labeled_source_bs, num_workers=params.num_workers)

    if params.us:
        print('Using source data {} (unlabeled)'.format(params.source_dataset))
        us = get_dataloader(dataset_name=params.source_dataset, augmentation=params.augmentation,
                            batch_size=unlabeled_source_bs, num_workers=params.num_workers,
                            siamese=True)  # important

    if params.ut:
        print('Using target data {} (unlabeled)'.format(params.target_dataset))
        # if params.target_dataset == 'ISIC':
        #     transform = ISIC_few_shot.TransformLoader(224).get_composed_transform(aug=True)
        #     # transform_test = ISIC_few_shot.TransformLoader(
        #     #     args.image_size).get_composed_transform(aug=False)
        #     dataset = ISIC_few_shot.SimpleDataset(transform, split="datasets/split_seed_1/ISIC_unlabeled_20.csv")
        #     ind = torch.randperm(len(dataset))
        #     train_ind = ind[:int(0.9 * len(ind))]
        #     # val_ind = ind[int(0.9 * len(ind)):]
        #     trainset = torch.utils.data.Subset(dataset, train_ind)
        #     ut = torch.utils.data.DataLoader(trainset, batch_size=unlabeled_target_bs,
        #                                               num_workers=params.num_workers,
        #                                               shuffle=True, drop_last=True)
        ut = get_unlabeled_dataloader(dataset_name=params.target_dataset, augmentation=params.augmentation,
                                      batch_size=unlabeled_target_bs, num_workers=params.num_workers, siamese=True,
                                      unlabeled_ratio=params.unlabeled_ratio)

    return ls, us, ut

def L2_sp(model, model_ref):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sum = 0
    for k, (m1, m2) in enumerate(zip(model.modules(), model_ref.modules())):
        if isinstance(m1, nn.Conv2d):
            L2 = torch.norm(m1.weight.data.to(device) - m2.weight.data.to(device), p=2)
            sum += L2
    return sum

def frozee_model(model, k):
    names = []
    for n, m in model.named_parameters():
        if m.requires_grad:
            names.append(n)
    names_sub = names[k:]
    for n, m in model.named_parameters():
        if n in names_sub:
            m.requires_grad = False

def pruning_layer(model, model2, prune_rate):
    for k, (m1, m2) in enumerate(zip(model.modules(), model2.modules())):
        if isinstance(m1, nn.Conv2d):
            size = m1.weight.data.numel()
            weight_abs = m1.weight.data.view(-1).abs().clone()
            y, _ = torch.sort(weight_abs)
            index = int(size * prune_rate)
            thre = y[index]
            weight_copy = m1.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            m2.weight.data.mul_(mask)
    return model2.state_dict()



def pr(epoch, total_epoch, p):
    prune_rate = p * epoch/total_epoch
    return prune_rate

def sparse_grad(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m.weight.data.mul_(mask)


def main(params):
    tic = time.time()

    if params.model == 'swav':
        import backbone
        model_dict = {params.backbone: backbone.ResNet10(method=params.method, track_bn=params.track_bn,
                                                      reinit_bn_stats=params.reinit_bn_stats)}
        baseline = BaselineTrain(model_dict[params.backbone], num_class=params.num_classes)
        model = SwAV(baseline)
    else:
        backbone = get_backbone_class(params.backbone)()
        model = get_model_class(params.model)(backbone, params)
        model_ref = get_model_class(params.model)(backbone, params)
    output_dir = get_output_directory(params)
    labeled_source_loader, unlabeled_source_loader, unlabeled_target_loader = _get_dataloaders(params)

    params_path = get_pretrain_params_path(output_dir)
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)
    pretrain_history_path = get_pretrain_history_path(output_dir)
    print('Saving pretrain params to {}'.format(params_path))
    print('Saving pretrain history to {}'.format(pretrain_history_path))

    if params.pls or params.pmsl:
        assert (params.pls and params.pmsl) is False
        # Load previous pre-trained weights for second-step pre-training
        previous_base_output_dir = get_output_directory(params, pls_previous=params.pls, pmsl_previous=params.pmsl)
        state_path = get_final_pretrain_state_path(previous_base_output_dir)
        if params.model == 'swav':
            state_path = './logs/output/mini/resnet10_swav_LS_default/pretrain_state_1000.pt'
        print('Loading previous state for second-step pre-training:')
        print(state_path)

        # Note, override model.load_state_dict to change this behavior.
        state = torch.load(state_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        init_dict = model.state_dict()
        model_ref.load_state_dict(init_dict)
        if len(unexpected):
            raise Exception("Unexpected keys from previous state: {}".format(unexpected))
    elif params.imagenet_pretrained:
        print("Loading ImageNet pretrained weights")
        backbone.load_imagenet_weights()

    model.train()
    model.cuda()

    if params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params.lr, momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=False)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        raise ValueError('Invalid value for params.optimizer: {}'.format(params.optimizer))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[400, 600, 800],
                                                     gamma=0.1)

    pretrain_history = {
        'loss': [0] * params.epochs,
        'source_loss': [0] * params.epochs,
        'target_loss': [0] * params.epochs,
    }

    for epoch in range(params.epochs):
        print('EPOCH {}'.format(epoch).center(40).center(80, '#'))

        epoch_loss = 0
        epoch_source_loss = 0
        epoch_target_loss = 0
        steps = 0
        if epoch>0 and (epoch+1) % params.fre == 0 and params.pls:
            # p = pr(epoch+1, params.epochs, params.pr)
            model_ref.cuda()
            model.load_state_dict(pruning_layer(model, model_ref, params.pr))
            # pruning_layer(model, p)

        if epoch == 0:
            state_path = get_pretrain_state_path_pr(output_dir, params, epoch=0)
            print('Saving pre-train state to:')
            print(state_path)
            torch.save(model.state_dict(), state_path)

        model.on_epoch_start()
        model.train()

        if params.ne:
            projection = nn.Sequential(
                nn.Linear(512, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128),
            )
            projection = projection.cuda()
            projection_opt = torch.optim.SGD(projection.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                             weight_decay=0.001)

        if params.ls and not params.us and not params.ut:  # only ls (type 1)
            for x, y in tqdm(labeled_source_loader):
                model.on_step_start()
                optimizer.zero_grad()
                if params.model == 'swav':
                    loss = model.compute_cls_loss_and_accuracy(x.cuda(), y.cuda())
                else:
                    loss, _ = model.compute_cls_loss_and_accuracy(x.cuda(), y.cuda())
                loss.backward()
                optimizer.step()
                model.on_step_end()

                epoch_loss += loss.item()
                epoch_source_loss += loss.item()
                steps += 1
        elif not params.ls and params.us and not params.ut:  # only us (type 2)
            for x, _ in tqdm(unlabeled_source_loader):
                model.on_step_start()
                optimizer.zero_grad()
                loss = model.compute_ssl_loss(x[0].cuda(), x[1].cuda())
                loss.backward()
                optimizer.step()
                model.on_step_end()

                epoch_loss += loss.item()
                epoch_source_loss += loss.item()
                steps += 1
        elif params.ut:
            # ut (epoch is based on unlabeled target)
            # if params.pls:
            #     init_dict = model.state_dict()
            #     model_ref.load_state_dict(init_dict)
            for x, _ in tqdm(unlabeled_target_loader):
                model.on_step_start()
                optimizer.zero_grad()
                if params.model == 'swav':
                    target_loss = model.compute_ssl_loss(x)
                else:
                    target_loss = model.compute_ssl_loss(x[0].cuda(), x[1].cuda())  # UT loss
                epoch_target_loss += target_loss.item()
                source_loss = None
                if params.ls:  # type 4, 7
                    try:
                        sx, sy = labeled_source_loader_iter.next()
                    except (StopIteration, NameError):
                        labeled_source_loader_iter = iter(labeled_source_loader)
                        sx, sy = labeled_source_loader_iter.next()
                    if params.model == 'swav':
                        source_loss = model.compute_cls_loss_and_accuracy(sx.cuda(), sy.cuda())
                    else:
                        source_loss = model.compute_cls_loss_and_accuracy(sx.cuda(), sy.cuda())[0]  # LS loss
                    epoch_source_loss += source_loss.item()
                if params.us:  # type 5, 8
                    try:
                        sx, sy = unlabeled_source_loader_iter.next()
                    except (StopIteration, NameError):
                        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
                        sx, sy = unlabeled_source_loader_iter.next()
                    if params.model == 'swav':
                        source_loss = model.compute_ssl_loss(sx.cuda())
                    else:
                        source_loss = model.compute_ssl_loss(sx[0].cuda(), sx[1].cuda())  # US loss
                    epoch_source_loss += source_loss.item()

                if params.ne:
                    xs, ys = labeled_source_loader_iter.next()
                    xs = xs.cuda()
                    if params.model == 'swav':
                        scores_s = model.base_encoder(xs)
                        scores_t = model.base_encoder(x[1].cuda())
                    else:
                        scores_s = model.backbone(xs)
                        scores_t = model.backbone(x[1].cuda())
                    z_s = projection(scores_s)
                    z_t = projection(scores_t)
                    z_s = z_s.detach()
                    z_s = F.normalize(z_s)
                    z_t = F.normalize(z_t)
                    loss_ne = (z_s * z_t).sum(dim=1).mean()


                if params.ne:
                    if source_loss:
                        loss = source_loss * (1 - params.gamma) + target_loss * params.gamma-0.01*loss_ne

                    else:
                        loss = target_loss+0.1*loss_ne
                else:
                    if source_loss:
                        loss = source_loss * (1 - params.gamma) + target_loss * params.gamma

                    else:
                        loss = target_loss
                if params.model == 'swav':
                    if epoch < 313:
                        for name, p in model.named_parameters():
                            if "prototypes" in name:
                                p.grad = None
                # frozee_model(model, 10)
                loss = loss + params.alpha*L2_sp(model, model_ref)
                loss.backward()
#                if params.pls:
#                    sparse_grad(model)
                optimizer.step()
                # for m, (n1, n2) in enumerate(zip(model.modules(), model_ref.modules())):
                #     if isinstance(n1, nn.Conv2d):
                #         print(m)
                #         print(n1.weight.data)
                #         print(n2.weight.data)

                if params.ne:
                    projection_opt.step()
                model.on_step_end()

                epoch_loss += loss.item()
                steps += 1
        else:
            raise AssertionError('Unknown training combination.')

        if scheduler is not None:
            scheduler.step()
        model.on_epoch_end()

        mean_loss = epoch_loss / steps
        mean_source_loss = epoch_source_loss / steps
        mean_target_loss = epoch_target_loss / steps
        fmt = 'Epoch {:04d}: loss={:6.4f} source_loss={:6.4f} target_loss={:6.4f}'
        print(fmt.format(epoch, mean_loss, mean_source_loss, mean_target_loss))

        pretrain_history['loss'][epoch] = mean_loss
        pretrain_history['source_loss'][epoch] = mean_source_loss
        pretrain_history['target_loss'][epoch] = mean_target_loss

        pd.DataFrame(pretrain_history).to_csv(pretrain_history_path)

        epoch += 1
        if epoch % params.model_save_interval == 0 or epoch == params.epochs:
            state_path = get_pretrain_state_path_pr(output_dir, params, epoch=epoch)
            print('Saving pre-train state to:')
            print(state_path)
            torch.save(model.state_dict(), state_path)


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('pretrain')
    tic = time.perf_counter()
    targets = params.target_dataset
    if targets is None:
        targets = [targets]
    elif len(targets) > 1:
        print('#' * 80)
        print("Running pretrain iteratively for multiple target datasets: {}".format(targets))
        print('#' * 80)
    
    for target in targets:
        params.target_dataset = target
        main(params)
    toc = time.perf_counter()
    runtime = toc-tic
    print("运行时间：", runtime)
