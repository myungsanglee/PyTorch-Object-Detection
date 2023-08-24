# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils customized by Myungsang Lee
"""
import os
import argparse
import random
from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from yaml_helper import get_configs

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))


def get_labels_shapes(img_path):
    # verify images
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    assert (w > 9) & (h > 9), f'image size ({w}, {h}) <10 pixels'
    assert img is not None, f'invalid image file {img_path}'
    
    # verify labels
    txt_path = img_path.rsplit('.', 1)[0] + '.txt'
    with open(txt_path, 'r') as f:
        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
        lb = np.array(label, dtype=np.float32)
    nl = len(lb)
    if nl:
        assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
        assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
        assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
            lb = lb[i]  # remove duplicates
            print(f'WARNING ⚠️ {img_path}: {nl - len(i)} duplicate labels removed')
    else:
        lb = np.zeros((0, 5), dtype=np.float32)

    return lb, [w, h]


def check_anchors(cfg, thr=4.0):
    # get train dataset labels & shapes
    dataset_labels = []
    dataset_shapes = []
    
    with open(cfg['train_list'], 'r') as f:
        img_file_list = f.read().splitlines()
    
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(pool.imap(get_labels_shapes, img_file_list), total=len(img_file_list))
        for label, shape in pbar:
            dataset_labels.append(label)
            dataset_shapes.append(shape)
    
    dataset_shapes = np.array(dataset_shapes, dtype=np.float32)
    input_size = cfg['input_size']
    
    # Check anchor fit to data, recompute if necessary
    shapes = input_size * dataset_shapes / dataset_shapes.max(1, keepdims=True)
    # scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    # wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset_labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = torch.tensor(cfg['anchors'], dtype=torch.float32)  # current anchors
    bpr, aat = metric(anchors)
    s = f'\n{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        print(f'{s}Current anchors are a good fit to dataset ✅')
        anchors = np.array(anchors, dtype=np.float32)
    else:
        print(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        na = anchors.size(0)  # number of anchors
        anchors = kmean_anchors(dataset_labels, dataset_shapes, n=na, img_size=input_size, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr >= bpr:  # replace anchors
            s = f'Done ✅ '
        else:
            s = f'Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
            anchors = np.array(cfg['anchors'], dtype=np.float32)
        print(s)

    s = '\n[Anchors]\n'
    for x in anchors:
        s += '%i,%i, ' % (round(x[0]), round(x[1]))
    print(s[:-2])


def kmean_anchors(dataset_labels, dataset_shapes, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset_labels: a loaded dataset labels
            dataset_shapes: a loaded dataset shapes
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean:\n'
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            print(s[:-2])
        return k

    # Get label wh
    shapes = img_size * dataset_shapes / dataset_shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset_labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        print(f'Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        print(f'WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen))  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)
    
    check_anchors(cfg)