import logging
logger = logging.getLogger(__name__)

from zfold.dataset.scripts.bak.pkl_dataset import *
from zfold.utils.gen_lbl import convert_DisOri_label_v2

#TODO
tpl_num = 4
mask_index = -1

class NPZDataset(torch.utils.data.Dataset):
    """
    For loading protein sequence datasets in the common FASTA data format
    """
    def __init__(self, path,
                       semi_path = None,
                       ratio = 0.25,
                       num = None,
                       is_train = False,
                       tpl_mode = 'v1'):

        self.num = num
        self.is_train = is_train
        self.tpl_mode = tpl_mode

        logger.info(f'Supervised MSA dataset : {path}')
        logger.info(f'Semi-Supervised MSA dataset : {semi_path}')

        def process(path, num = None):
            pdb_ids = []
            msanpzs, lbls, tpls, tplnpzs = [],[],[],[]

            for id_file, msanpz_dir, lbl_dir, tpl_dir, tplnpz_dir in path:
                f = open(id_file, 'r')
                samples = f.readlines()
                f.close()

                pdb_ids_ = [sample.strip() for sample in samples]
                msanpzs_ = [None for pdb_id in pdb_ids_]
                lbls_ = [None for pdb_id in pdb_ids_]
                tpls_ = [None for pdb_id in pdb_ids_]
                tplnpzs_ = [None for pdb_id in pdb_ids_]

                if msanpz_dir is not None:
                    msanpzs_ = [f'{msanpz_dir}/{pdb_id}.npz' for pdb_id in pdb_ids_]
                if lbl_dir is not None:
                    lbls_ = [f'{lbl_dir}/{pdb_id}.npy' for pdb_id in pdb_ids_]
                if tpl_dir is not None:
                    tpls_ = [f'{tpl_dir}/{pdb_id}/template.pkl' for pdb_id in pdb_ids_]
                if tplnpz_dir is not None:
                    tplnpzs_ = [f'{tplnpz_dir}/{pdb_id}.npz' for pdb_id in pdb_ids_]

                pdb_ids.extend(pdb_ids_)
                msanpzs.extend(msanpzs_)
                lbls.extend(lbls_)
                tpls.extend(tpls_)
                tplnpzs.extend(tplnpzs_)

            data = list(zip(pdb_ids, msanpzs, lbls, tpls, tplnpzs))
            random.shuffle(data)

            if num is not None:
                while len(data) < num:
                    data.extend(data)
                random.shuffle(data)
                data = data[:num]

            return data

        data = process(path, num)

        logger.info(f'Supervised MSA dataset - num: {len(data)}')

        if semi_path is not None:
            num_semi = int(ratio / (1 - ratio) * len(data))
            logger.info(f'Semi-Supervised Ratio: {ratio}, Semi-Supervised MSA dataset - num: {num_semi}')
            data_semi = process(semi_path, num_semi)
            data.extend(data_semi)
            random.shuffle(data_semi)

        self.pdb_ids = [item[0] for item in data]
        self.msanpzs = [item[1] for item in data]
        self.lbls = [item[2] for item in data]
        self.tpls = [item[3] for item in data]
        self.tplnpzs = [item[4] for item in data]

        self.num = len(self.pdb_ids)
        logger.info(f'ALL MSA dataset - num: {self.num}')

    def __getitem__(self, idx):
        idx = idx % len(self.pdb_ids)
        return self.pdb_ids[idx], \
               self.msanpzs[idx], \
               self.tpls[idx], \
               self.tplnpzs[idx], \
               self.lbls[idx]

    def __len__(self):
        return self.num

msa_keys = ['msa_mask', 'msa_row_mask', 'true_msa', 'msa_feat', 'msa_esm_tokens', 'true_msa_esm_tokens']
extra_msa_keys = ['extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'extra_has_deletion', 'extra_deletion_value', 'extra_msa_esm_tokens']

class EncodedNPZDataset(NPZDataset):
    """
    The FastaDataset returns raw sequences - this allows us to return
    indices with a dictionary instead.
    """
    def __init__(self,
                 path,
                 semi_path = None,
                 ratio = 0.25,
                 is_train = False,
                 crop_size = 128,
                 msa_depth = 128,
                 extra_msa_depth = 256,
                 bins = [37, 25, 25, 25],
                 tpl_mode = 'v1',
                 num = None,
                 ):

        super().__init__(path,
                         semi_path = semi_path,
                         ratio = ratio,
                         is_train = is_train,
                         tpl_mode = tpl_mode,
                         num = num)

        self.alphabet = esm.Alphabet.from_architecture('MSA Transformer')
        self.pad_idx = self.alphabet.pad()
        self.bins = bins
        self.tpl_mode = tpl_mode
        self.sizes = np.asarray([crop_size for i in range(self.num)])

        self.msa_depth = msa_depth
        self.crop_size = crop_size
        self.extra_msa_depth = extra_msa_depth

        self.msa_mask_prob = 0.15
        self.subsample = True

    def __getitem__(self, idx):
        try:
            pdb_id, msanpz, tpl, tplnpz, lbl = super().__getitem__(idx)

            #1. MSA tokens
            processed_feature_dict = parse_msanpz(msanpz)

            seq_len = processed_feature_dict['aatype'].shape[0]

            #1.1 subsample msa & extra msa
            msa_index = [i+1 for i in range(processed_feature_dict['true_msa_esm_tokens'].shape[0]-1)]
            extra_msa_index = [i+1 for i in range(processed_feature_dict['extra_msa_esm_tokens'].shape[0]-1)]

            if self.subsample and self.is_train:
                random.shuffle(msa_index)
                random.shuffle(extra_msa_index)

            msa_index = np.asarray([0] + msa_index[:self.msa_depth - 1])
            extra_msa_index = np.asarray([0] + extra_msa_index[:self.extra_msa_depth -1])

            for key in msa_keys:
                processed_feature_dict[key] = processed_feature_dict[key][msa_index, ...]

            for key in extra_msa_keys:
                processed_feature_dict[key] = processed_feature_dict[key][extra_msa_index, ...]

            #1.2 apply random masks on MSA tokens
            msa_masks = (np.random.uniform(size=processed_feature_dict['true_msa_esm_tokens'].shape) <= self.msa_mask_prob).astype(np.int8)
            processed_feature_dict['masked_msa_esm_tokens'] = processed_feature_dict['true_msa_esm_tokens'].clone()

            if self.is_train:
                processed_feature_dict['masked_msa_esm_tokens'][msa_masks == 1] = self.alphabet.mask_idx

            #2. Process template related features
            if not os.path.exists(tpl):
                tgt = os.path.basename(os.path.split(tpl)[0])
                tpl = f'{os.path.split(os.path.split(tpl)[0])[0]}/{tgt}.pkl'

            if not os.path.exists(tpl):
                tpl = None
                print(f'{tpl} does not exists')
            if not os.path.exists(tplnpz):
                tplnpz = None
                print(f'{tplnpz} does not exists')

            tinputs = None if (tpl is None and tplnpz is None) else [tpl, tplnpz]

            feat_dict = parse_tpl_file(path = tinputs,
                                       n_resds = seq_len,
                                       tpl_topk = 4,
                                       is_train = self.is_train,
                                       mode = self.tpl_mode)

            processed_feature_dict.update(feat_dict)  # structural templates

            #3. Distance&Orientation labels
            label = np.load(lbl)

            lbl, agnle0, angle1, angle2 = convert_DisOri_label_v2(label, np.ones_like(label), bins=self.bins)

            lbls = torch.cat([torch.LongTensor(lbl).unsqueeze(0),
                              torch.LongTensor(agnle0).unsqueeze(0),
                              torch.LongTensor(angle1).unsqueeze(0),
                              torch.LongTensor(angle2).unsqueeze(0)], dim=0)

            processed_feature_dict['lbls'] = lbls

            processed_feature_dict = crop_feature_dict(processed_feature_dict, self.crop_size, is_train=self.is_train)

            return processed_feature_dict

        except:
            print('sth is wrong, return first sample')
            return self.__getitem__(0)
