import os
import numpy as np
import random
import torch

PI = 3.141592653

def convert_DisOri_label_v2(label_, mask_, bins = [37, 25, 25, 25], file = 'tmp.png'):
    """Get distanace&orientation ground truth labels."""

    #distance label CB
    mask = (mask_[:, :, 1] == 0)
    label_dis_CB = label_[:, :, 1]
    label_dis_cls_CB = np.zeros(label_dis_CB.shape)

    bins_cb = bins[0] - 1
    assert bins_cb % 18 == 0
    dist_inter = 18 / bins_cb
    for i in range(bins_cb):
        s = 2.0 + dist_inter * i
        e = 2.0 + dist_inter * (i + 1)
        label_dis_cls_CB[(label_dis_CB > s) & (label_dis_CB < e)] = i + 1
    label_dis_cls_CB[mask] = -1

    def get_angle_cls(label_angle, mask_angle, bin=24):
        a_min, a_max = -PI, PI

        bin_inter = 2 * PI / bin
        label_angle_cls = np.ones(label_angle.shape) * (-1)

        for i in range(bin):
            s = a_min + i * bin_inter
            e = a_min + (i+1) * bin_inter
            label_angle_cls[(label_angle >= s) & (label_angle < e)] = i + 1

        # -1 is masked label
        label_angle_cls[mask_angle == 0] = -1

        # 0 is non-contact bin for distance and orientation
        label_angle_cls[label_dis_cls_CB == 0] = 0

        return label_angle_cls

    #3.orientation labels, omega, theta, phi
    label_angle_cls0   = get_angle_cls(label_[:, :, 2], mask_[:, :, 2], bins[1]-1)
    label_angle_cls1   = get_angle_cls(label_[:, :, 3], mask_[:, :, 3], bins[2]-1)
    label_angle_cls2   = get_angle_cls(label_[:, :, 4], mask_[:, :, 4], bins[3]-1)

    # for debug
    debug = False
    if debug:
        plt.figure(figsize=(64, 64))

        labls_dcls = label_dis_cls_CA
        plt.subplot(1, 8, 1)
        plt.imshow(labls_dcls)
        plt.title('label dis cls argmax CA')

        labls_dcls = label_dis_cls_CB
        plt.subplot(1, 8, 2)
        plt.imshow(labls_dcls)
        plt.title('label dis cls argmax CB')

        labls_omega = label_angle_cls0
        labls_omega[labls_omega == -1] = 0
        plt.subplot(1, 8, 3)
        plt.imshow(labls_omega)
        plt.title('omega label')

        labls_theta = label_angle_cls1
        labls_theta[labls_theta == -1] = 0
        plt.subplot(1, 8, 4)
        plt.imshow(labls_theta)
        plt.title('theta label')

        labls_gamma = label_angle_cls2
        plt.subplot(1, 8, 5)
        plt.imshow(labls_gamma)
        plt.title('phi label')

        plt.savefig(file)
        plt.close()
        exit(0)

    return label_dis_cls_CB, \
           label_angle_cls0,\
           label_angle_cls1,\
           label_angle_cls2

def delta(label_delta, bin_num = 20):
    """Get delta cls labels."""
    #1.delta label
    label_delta_cls = np.zeros(label_delta.shape)
    for i in range(bin_num):
        s = (20.0 / bin_num) * i - 10.0
        e = (20.0 / bin_num) * (i + 1) - 10.0
        label_delta_cls[(label_delta >= s) & (label_delta < e)] = i + 1

    return label_delta_cls
