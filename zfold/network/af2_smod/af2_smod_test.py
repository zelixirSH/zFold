"""Unit-tests for the <AF2SMod> module."""

import os
import random
import logging

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zfold.dataset.utils import zfold_init
from zfold.network.af2_smod.utils import calc_rot_n_tsl
from zfold.network.af2_smod.utils import apply_trans
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.network.af2_smod.af2_smod import AF2SMod


def parse_prot(fas_fpath, pdb_fpath, npz_fpath, device):
    """Parse the protein data to obtain the native structure & MSA/pair features."""

    # parse the PDB file
    aa_seq, cord_tns, mask_mat, error_msg = ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath)
    cord_tns = cord_tns.to(device)
    mask_mat = mask_mat.to(device)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath

    # parse the NPZ file
    with np.load(npz_fpath) as npz_data:
        sfea_tns_np = npz_data['mfea'][0]  # L x D_s
        pfea_tns_np = npz_data['pfea']  # L x L x D_p
    sfea_tns = torch.tensor(sfea_tns_np, dtype=torch.float32, device=device).unsqueeze(dim=0)
    pfea_tns = torch.tensor(pfea_tns_np, dtype=torch.float32, device=device).unsqueeze(dim=0)

    return aa_seq, cord_tns, mask_mat, sfea_tns, pfea_tns


def calc_loss(helper, params_list):
    """Calculate the loss function."""

    # initialization
    n_lyrs = len(params_list)
    n_resds = params_list[0][0].shape[1]

    # calculate the loss function for each set of QTA parameters
    loss_list = []
    metrics_finl = {}
    cord_tns_list = []
    for idx_lyr, (quat_tns, trsl_tns, angl_tns) in enumerate(params_list):
        params = {
            'quat': quat_tns.view(n_resds, 3),
            'trsl': trsl_tns.view(n_resds, 3),
            'angl': angl_tns.view(n_resds, -1, 2),
        }
        atom_set = 'ca' if idx_lyr != n_lyrs - 1 else 'fa'
        loss, metrics, cord_tns = helper.calc_loss(params, atom_set, rtn_cord=True)
        loss_list.append(loss)
        metrics_finl['dRMSD-CA-L%d' % (idx_lyr + 1)] = metrics['dRMSD-CA']
        metrics_finl['dRMSD-BB-L%d' % (idx_lyr + 1)] = metrics['dRMSD-BB']
        metrics_finl['dRMSD-FA-L%d' % (idx_lyr + 1)] = metrics['dRMSD-FA']
        cord_tns_list.append(cord_tns)

    # calculate the overall loss function
    loss = torch.sum(torch.stack(loss_list))

    return loss, metrics_finl, cord_tns_list


def train(prot_ids, config):
    """Train a <AF2SMod> module."""

    # build the list of protein data
    prot_data_list = []
    for prot_id in prot_ids:
        # obtain the native structure & MSA/pair features
        fas_fpath = os.path.join(config['fas_dpath'], '%s.fasta' % prot_id)
        pdb_fpath = os.path.join(config['pdb_dpath'], '%s.pdb' % prot_id)
        npz_fpath = os.path.join(config['npz_dpath'], '%s.npz' % prot_id)
        aa_seq, cord_tns, mask_mat, sfea_tns, pfea_tns = \
            parse_prot(fas_fpath, pdb_fpath, npz_fpath, config['device'])

        # build the protein data from randomly transformed native structure(s)
        n_resds, n_atoms, _ = cord_tns.shape
        for idx in range(config['n_ornts']):
            # apply a global transformation on the native structure
            x1 = 10.0 * torch.randn((3), dtype=torch.float32, device=config['device'])
            x2 = 10.0 * torch.randn((3), dtype=torch.float32, device=config['device'])
            x3 = 10.0 * torch.randn((3), dtype=torch.float32, device=config['device'])
            rot_mat, tsl_vec = calc_rot_n_tsl(x1, x2, x3)
            cord_tns_new = apply_trans(cord_tns, rot_mat, tsl_vec).view(n_resds, n_atoms, 3)

            # perform the idealization-reconstruction process
            helper = FapeLossHelper()
            helper.preprocess(aa_seq, cord_tns_new, mask_mat)
            pdb_fpath = os.path.join(config['out_dpath'], '%s_native_%02d.pdb' % (prot_id, idx))
            ProtStruct.save(aa_seq, helper.cord_tns, helper.cmsk_mat, pdb_fpath)

            # record the protein data
            prot_data = {
                'id': prot_id,
                'seq': aa_seq,
                'helper': helper,
                'sfea': sfea_tns,
                'pfea': pfea_tns,
            }
            prot_data_list.append(prot_data)

    # create the model, optimizer, and LR scheduler
    model = AF2SMod(
        n_lyrs=config['n_lyrs'],
        n_dims_sfea=config['n_dims_sfea'],
        n_dims_pfea=config['n_dims_pfea'],
    ).to(config['device'])
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[256, 384], gamma=0.1)
    #scheduler = ReduceLROnPlateau(
    #    optimizer, mode='min', factor=0.5, patience=8, cooldown=8, min_lr=1e-6, verbose=True)

    # optimize the <AF2SMod> module to minimize the FAPE loss
    prot_data_list *= 4
    for idx_iter in range(config['n_iters']):
        loss_list = []
        random.shuffle(prot_data_list)
        for prot_data in prot_data_list:
            params_list = model(prot_data['sfea'], prot_data['pfea'])
            loss, metrics, _ = calc_loss(prot_data['helper'], params_list)
            loss_list.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            drmsd = [metrics['dRMSD-CA-L%d' % (x + 1)] for x in range(config['n_lyrs'])]
            logging.info('%s: dRMSD = %s', prot_data['id'], ','.join(['%.4f' % x for x in drmsd]))
        loss = torch.mean(torch.stack(loss_list))
        logging.info('iter #%d: FAPE = %.4f', idx_iter + 1, loss.item())
        scheduler.step()
        #scheduler.step(loss.item())

    # evaluate the final <AF2SMod> module
    for prot_data in prot_data_list:
        logging.info('evaluating the <AF2SMod> module on <%s>', prot_data['id'])
        params_list = model(prot_data['sfea'], prot_data['pfea'])
        loss, metrics, cord_tns_list = calc_loss(prot_data['helper'], params_list)
        logging.info('FAPE loss: %.4f', loss.item())
        for key, val in metrics.items():
            logging.info('%s: %.4f', key, val)
        for idx_lyr, cord_tns in enumerate(cord_tns_list):
            pdb_fpath = os.path.join(config['out_dpath'], '%s_decoy_%02d.pdb' % (prot_data['id'], idx_lyr + 1))
            ProtStruct.save(prot_data['seq'], cord_tns, prot_data['helper'].cmsk_mat, pdb_fpath)


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    config = {
        'n_iters': 512,
        'n_lyrs': 2,
        'n_dims_sfea': 384,
        'n_dims_pfea': 256,
        'device': torch.device('cuda:0'),
        'fas_dpath': os.path.join(curr_dir, 'casp14/fasta.files'),
        'pdb_dpath': os.path.join(curr_dir, 'casp14/pdb.files.native'),
        'npz_dpath': os.path.join(curr_dir, 'casp14/npz.files.xfold'),
        'out_dpath': os.path.join(curr_dir, 'outputs.af2_smod'),
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # train a <AF2SMod> module - single protein, single orientation
    prot_ids = ['T1082-D1']
    config['n_ornts'] = 1  # number of orientations per protein
    train(prot_ids, config)

    # train a <AF2SMod> module - single protein, multiple orientations
    #prot_ids = ['T1082-D1']
    #config['n_ornts'] = 4  # number of orientations per protein
    #train(prot_ids, config)

    # train a <AF2SMod> module - multiple proteins, single orientation per protein
    #prot_ids = ['T1082-D1', 'T1084-D1']
    #prot_ids = ['T1046s1-D1', 'T1070-D4', 'T1073-D1', 'T1082-D1', 'T1084-D1']
    #config['n_ornts'] = 4  # number of orientations per protein
    #train(prot_ids, config)


if __name__ == '__main__':
    main()
