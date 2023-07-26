import numpy as np
import jittor as jt
from jittor import nn
from skimage.exposure import match_histograms


def getHistMatched(imgs: jt.Var, refs: jt.Var):
    assert imgs.shape == refs.shape
    bs, c, h, w = imgs.shape
    matched_list = []
    imgs, refs = imgs.clamp(0, 1).data, refs.clamp(0, 1).data
    print(imgs.shape)
    for i in range(bs):
        img, ref = imgs[i, :, :, :], refs[i, :, :, :]
        img = np.array((img + 1) / 2 * 255, dtype=np.uint8).transpose((1, 2, 0))
        ref = np.array((ref + 1) / 2 * 255, dtype=np.uint8).transpose((1, 2, 0))
        matched = match_histograms(img, ref, channel_axis=-1)
        matched = matched.transpose((2, 0, 1)) / 255
        matched = matched.reshape(1, *matched.shape)
        matched_list.append(matched)
    matcheds = jt.array(np.concatenate(matched_list, axis=0), dtype="float32").stop_grad()
    return matcheds


def HistLoss(imgs: jt.Var, refs: jt.Var):
    matcheds = getHistMatched(imgs, refs)
    return nn.L1Loss()(imgs, matcheds)
