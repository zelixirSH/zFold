try:
    import matplotlib.pyplot as plt
except:
    print('matplotlib import error')

import random
import numpy as np
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

########################################################################################################################
def get_rnd_crop_indices( seq_len, crop_size, enbl_crop = True):
    """Get 1-D indices for angle prediction or 2-D indices for distance/contact prediction."""
    if not enbl_crop:
        indices = None
    else:
        maxval = seq_len - crop_size
        if maxval <= 0:
            return None
        idx_hrz = random.randint(0,maxval)
        idx_vrt = random.randint(0,maxval)
        indices = [idx_hrz, idx_vrt]

    return indices

def get_rnd_sample_indices(seq_len, sample_size):
    """Get 1-D indices for angle prediction or 2-D indices for distance/contact prediction."""
    tmp = np.arange(seq_len)
    random.shuffle(tmp)
    index_h = np.sort(tmp[:sample_size])
    random.shuffle(tmp)
    index_v = np.sort(tmp[:sample_size])
    return [index_h, index_v]
########################################################################################################################
def save_npz_png(npz, save_path):
    data = dict(np.load(npz, allow_pickle=True))
    save_png(data['dist'], data['omega'], data['theta'], data['phi'],save_path=save_path)

def save_png(logit_dist, logit_omega = None, logit_theta = None, logit_psi = None, save_path = 'tmp.png'):
    npz = logit_dist[:, :, 5:]
    DSTEP = 0.5
    dist_vals_np = (4.25 + DSTEP * np.arange(32, dtype=np.float32))
    dist_vals = dist_vals_np.reshape([1, 1, 32])
    preds_reg = np.sum(dist_vals * npz, axis=2)

    preds_reg[preds_reg <= 5] = 0
    preds_reg[preds_reg >= 19] = 0

    plt.figure(figsize=(12, 9))
    plt.subplot(2, 3, 1)
    plt.imshow(20 - preds_reg)
    plt.title('Predicted Distance')

    logit_tmp_ = np.argmax(logit_dist, 2)
    plt.subplot(2, 3, 2)
    plt.imshow(logit_tmp_)
    plt.title('dis cls argmax')

    if logit_omega is not None:
        logit_omega_tmp_ = np.argmax(logit_omega, 2)
        plt.subplot(2, 3, 4)
        plt.imshow(logit_omega_tmp_)
        plt.title('omega cls argmax')

    if logit_theta is not None:
        logit_theta_tmp_ = np.argmax(logit_theta, 2)
        plt.subplot(2, 3, 5)
        plt.imshow(logit_theta_tmp_)
        plt.title('theta cls argmax')

    if logit_psi is not None:
        logit_psi_tmp_ = np.argmax(logit_psi, 2)
        plt.subplot(2, 3, 6)
        plt.imshow(logit_psi_tmp_)
        plt.title('psi cls argmax')

    plt.savefig(save_path)
    plt.close()

    print('    save png to: '+ save_path)

def ave_af2_npz_list(npz_list, npz_out, is_png = True):
    # ave given npz_list
    logit_dists =  []

    for npz in npz_list:
        data = dict(np.load(npz, allow_pickle=True))
        logit_dists.append(data['logits'])

    logit_dist_ave = np.mean(logit_dists, axis=0)
    print(f'save npz {npz_out}')
    np.savez_compressed(npz_out, logits=logit_dist_ave)

def ave_trros_npz_list(npz_list, npz_out, is_png = True):
    # ave given npz_list
    logit_dists, logit_omegas, logit_thetas, logit_psis = [], [], [], []

    for npz in npz_list:
        data = dict(np.load(npz, allow_pickle=True))

        logit_dists.append(data['dist'])
        if 'omega' in data and 'theta' in data and 'phi' in data:
            logit_omegas.append(data['omega'])
            logit_thetas.append(data['theta'])
            logit_psis.append(data['phi'])
        else:
            print(f'orientations do not exist in {npz}')

    logit_dist_ave = np.mean(logit_dists, axis=0)

    npz = {'dist': logit_dist_ave}

    logit_omega_ave, logit_theta_ave, logit_psi_ave = None, None, None
    if len(logit_omegas) > 0 and  len(logit_thetas) > 0 and  len(logit_psis) > 0:
        logit_omega_ave = np.mean(logit_omegas, axis=0)
        logit_theta_ave = np.mean(logit_thetas, axis=0)
        logit_psi_ave = np.mean(logit_psis, axis=0)

        npz['omega'] = logit_omega_ave
        npz['theta'] = logit_theta_ave
        npz['phi'] = logit_psi_ave

    if is_png:
        save_png(logit_dist_ave, logit_omega_ave, logit_theta_ave, logit_psi_ave,
                     save_path=npz_out.replace('.npz', '.png'))

    np.savez_compressed(npz_out, **npz)



