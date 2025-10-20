import torch
import pickle
import random
import itertools
from Bio import SeqIO
from typing import List, Tuple
import string

from zfold.utils import *
from zfold.network import esm
from zfold.utils.gen_tpl_fea import *

try:
    import matplotlib.pyplot as plt
except:
    print('sth wrong with matplotlib')

def get_msa_feature(msa_path,
                    msa_data,
                    msa_depth,
                    batch_converter = esm.Alphabet.from_architecture('MSA Transformer').get_batch_converter(),
                    mode = 'TopN'):

    if msa_data is None:
        msa_data = [read_msa(msa_path, msa_depth, mode)]

    _, _, msa_batch_tokens = batch_converter(msa_data)

    #remove [cls] token in msa_batch_tokens
    fea1d = msa_batch_tokens.squeeze(0).data.cpu().numpy().transpose((1, 0))[1:, :]

    return torch.LongTensor(fea1d[:, :].transpose((1, 0)))

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename, min_len = 128, min_depth = 100, is_shuffle = True):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    num = 0
    seq_tmp = []
    seq_name = []

    for line in open(filename,"r"):
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seq_tmp += line.rstrip().translate(table)
        elif num == 0:
            num += 1
            seq_name.append(line.rstrip())
        else:
            seqs.append(''.join(seq_tmp))
            seq_tmp = []
            num += 1
            seq_name.append(line.rstrip())

    if len(seq_tmp) > 0:
        seqs.append(''.join(seq_tmp))

    assert len(seqs) == len(seq_name), print(len(seqs), len(seq_name))

    seqs_new, seq_name_new = [], []
    for sn, s in zip(seq_name, seqs):
        if '>ss_pred' == sn or '>ss_conf' == sn:
            continue
        seqs_new.append(s)
        seq_name_new.append(sn)

    seqs = seqs_new
    seq_name = seq_name_new

    seqs.append(''.join(seq_tmp))
    seq_len = len(seqs[0])

    X_ = ''.join(['X' for i in range(min_len)])
    seqs = [seq + X_ for seq in seqs]
    seqs = [seq.replace('-','X') for seq in seqs]
    seqs = [seq[:seq_len] for seq in seqs]

    while len(seqs) < min_depth:
        seqs += seqs

    seq_protein = seqs[0]
    if is_shuffle:
        random.shuffle(seqs)

    assert '-' not in seq_protein, print(seq_protein)
    return seqs, seq_len, seq_protein

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int, mode: string) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""

    if mode == 'TopN':
        return [(record.description, remove_insertions(str(record.seq))) for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

    elif mode == 'Rand':
        max_num = 10000
        all_seq = [(record.description, remove_insertions(str(record.seq))) for record in itertools.islice(SeqIO.parse(filename, "fasta"), max_num)]
        origin_seq = all_seq[:1]
        random.shuffle(all_seq)
        return origin_seq + all_seq[:nseq-1]

    else:
        raise RuntimeException(f'Mode not exists {mode}')


def get_features(msa_path,
                 tpl_path,
                 msa_depth = 100,
                 tpl_topk = 4,
                 is_train = True,
                 tpl_mode = 'v1',
                 mask_index = -1,
                 batch_converter_msa=esm.Alphabet.from_architecture('MSA Transformer').get_batch_converter(),
                 ):

    data = {}
    data['tokens_feats'] = None
    data['extra_tokens_feats'] = None

    if msa_path.endswith('.a3m'):
        msa_tokens = get_msa_feature(msa_path,
                                     msa_data = None,
                                     msa_depth = msa_depth,
                                     batch_converter = batch_converter_msa)
        data['msa_tokens'] = msa_tokens

    elif msa_path.endswith('.npz'):
        processed_feature_dict = parse_msanpz(msa_path)
        data['msa_tokens'] = processed_feature_dict['true_msa_esm_tokens']
        data['tokens_feats'] = None
        data['extra_tokens_feats'] = processed_feature_dict['extra_msa_esm_tokens']
    else:
        raise NotImplementedError

    t1ds, t2ds = get_tpl_feature(tpl_path,
                                 tpl_topk = tpl_topk,
                                 is_train = is_train,
                                 mask_index = mask_index,
                                 seq_len = data['msa_tokens'].shape[1],
                                 mode = tpl_mode)

    data['t1ds'] = t1ds
    data['t2ds'] = t2ds
    return data

def parse_msanpz(npz):

    processed_feature_dict = dict(np.load(npz))
    # convert true msa & masked msa tokens to esm tokens
    masked_msa_tokens = processed_feature_dict['msa_feat'][:, :, :23, 0]
    masked_msa_tokens = np.argmax(masked_msa_tokens, axis=-1)
    processed_feature_dict['msa_esm_tokens'] = af2idx_to_msatrans_val_arr[masked_msa_tokens]

    true_msa = processed_feature_dict['true_msa'][:, :, 0]
    processed_feature_dict['true_msa'] = true_msa
    processed_feature_dict['true_msa_esm_tokens'] = af2idx_to_msatrans_val_arr[true_msa]

    extra_msa = processed_feature_dict['extra_msa'][:, :, 0]
    processed_feature_dict['extra_msa'] = extra_msa
    processed_feature_dict['extra_msa_esm_tokens'] = af2idx_to_msatrans_val_arr[extra_msa]

    # Make sure the first seq in extra msa is the target sequence
    processed_feature_dict['extra_msa_esm_tokens'][0,:] = processed_feature_dict['true_msa'][0,:]

    """
    # # Apply random masks on true MSA tokens
    # msa_masks = (np.random.uniform(size=processed_feature_dict['true_msa_esm_tokens'].shape) <= mask_prob).astype(np.int8)
    # processed_feature_dict['masked_msa_esm_tokens'] = np.copy(processed_feature_dict['true_msa_esm_tokens'])
    #
    # if is_train:
    #     processed_feature_dict['masked_msa_esm_tokens'][msa_masks == 1] = mask_idx
    #
    # # pack into a dict
    # data_dict = {
    #     'msa-t': torch.tensor(msa_tokens_true, dtype=torch.int64),  # K x L
    #     'msa-p': torch.tensor(msa_tokens_pert, dtype=torch.int64),  # K x L
    #     'msa-m': torch.tensor(msa_masks, dtype=torch.int8),  # K x L
    # }  # K: number of homologous sequences
    """

    for key in processed_feature_dict.keys():
        processed_feature_dict[key] = torch.Tensor(processed_feature_dict[key]).type(type_dict[key])

    return processed_feature_dict


def sample_msa(msa_tokens_full, msa_depth, is_train, mask_prob=0.15, mode = 'random'):
    """Sample the MSA w/ optional random masks applied."""

    # initialization
    device = msa_tokens_full.device
    alphabet = esm.Alphabet.from_architecture('MSA Transformer')

    # sample a fixed number of sequences from the original MSA data
    n_seqs = msa_tokens_full.shape[1]
    if n_seqs <= msa_depth:
        msa_tokens_true = msa_tokens_full
    elif mode == 'topN':
        # use top-K' sequences during evaluation
        msa_tokens_true = msa_tokens_full[:, :msa_depth]
    elif mode == 'random':
        # randomly select K' sequences during training
        idxs_seq = [0] + list(random.sample(range(1, n_seqs), msa_depth - 1))
        msa_tokens_true = torch.stack([msa_tokens_full[:, x] for x in idxs_seq], dim=1)
    else:
        raise NotImplementedError

    # apply random masks on MSA tokens
    msa_tokens_pert = msa_tokens_true.detach().clone()
    msa_masks = (torch.rand(msa_tokens_pert.shape, device=device) < mask_prob).to(torch.int8)

    if is_train:
        msa_tokens_pert[msa_masks == 1] = alphabet.mask_idx

    return msa_tokens_true, msa_tokens_pert, msa_masks

if __name__ == '__main__':

    pkl_fpath = '/apdcephfs/share_1436367/seutao/RCSB-PDB-27k/fas.template/2MIO_A/template.pkl'

    print(pkl_fpath)

    feature_dict = load_obj(pkl_fpath)

    t1ds, t2ds = get_template_features(feature_dict, tpl_num=4, atom='CB', mask_index=-1, is_train=False)

    print(t1ds.shape, t2ds.shape)

    msa_path = f''
    tplnpz = f'/apdcephfs/share_1436367/seutao/RCSB-PDB-27k/files.tpl.npz/2MIO_A.npz'

    processed_feature_dict = dict(np.load(tplnpz))
    t1ds, t2ds = get_template_features_v3(processed_feature_dict, tpl_num=4, is_train=True)
    print(t1ds.shape, t2ds.shape)

    feature_dict.update(processed_feature_dict)
    t1ds, t2ds = get_template_features_v3(feature_dict, tpl_num=4, is_train=True, tpl_mode='v3.1')
    print(t1ds.shape, t2ds.shape)








