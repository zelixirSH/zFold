import ml_collections
from box import Box

XFOLD_CONFIG = ml_collections.ConfigDict(
    {
        # Activation
        'activation': 'relu',
        #
        'basic':{
                    'n_module': 1, # num of modules
                    'n_recycle': 3, # num of n_recycle
                    'd_msa': 384,
                    'd_pair': 256,
                    'n_layer_msa': 1, # num of MSA2MSA layer in each module
                    'n_layer_pair': 1, # num of Pair2Pair layer in each module
                    'n_head_msa': 12,
                    'n_head_pair': 8,
                    'r_ff': 4,
                    'msa_depth_max': 128,
                    'msa_mask_prob': 0.15,
        },
        #
        'modules':{
                    'msa2msa_type': 'MAB',
                    'pair2pair_type': 'PAB',
                    'pair2msa_type': 'bias_fusion',
                    'msa_emb': {
                        'is_onehot': False,
                        'd_msa': None,
                    },
                    'extra_msa_stack': {
                        'enable': False,
                        'random': False,
                        'recycle': False,
                        'is_onehot': True,
                        'd_msa': 25,
                        'dim': 128,
                    }
        },
        'attention':{
                    'n_layer_shift_tokens': 1,
                    'proj_type': 'linear', # 'DConv_project'
        },
        # template
        'templ':{
                    'use_templ': True,
                    'd_templ': 64,
                    'd_t1d': 23,
                    'd_t2d': 1,
                    'af2': False,
                    'af2_p_drop': 0.25,
                    'tpl_mode': 'v1',
        },
        # MSA bert
        'msa_bert':{
                    'msa_bert_config': None,
                    'skip_load_msa_bert': False,
                    'pos_emb': True,
        },
        # fas bert
        'fas_bert': {
                    'enable': False,
                    'local_dir': './facebook_esm_checkpoints',
        },
        # Position Embdedding
        'pos_emb':{
                    'is_pos_emb': False,
                    'is_peg': True,
                    'n_peg_block': 1,
                    'rpe_type': None,
        },
        # Structured dropout
        'dropout':{
                    'p_drop': 0.,
                    'p_attn_drop': 0.,
                    'p_layer_drop': 0.,
                    'is_p_layer_drop': True,
                    'TokenDropoutRowwise': 0.,
                    'TokenDropoutColumnwise': 0.,
        },
        # normalization
        'normalization':{
                    'is_use_ln': True,
                    'is_sandwich_norm': False,
                    'pb_relax': False,
        },
        # initialization
        'initialization':{
                    'is_rezero': False,
                    'is_init_zero': False,
        },
        #
        'dist':{
                    'is_dist_head': True,
                    'bins': [37, 25, 25, 25],
        },
        #
        'mlm': {
                    'is_lm_head':  True,
        },
        # AF2 structure module
        'af2_smod': {
                    'enable': False,
                    'n_lyrs': 8,
                    'n_dim_attn': 64,
                    'freeze': False,
                    'v2': True,
                    'detach_3d': False,
                    'finalloss': True,
                    'highres': False,
                    'plddt': False,
                    'g_recycle': 1,
        },
        # Optional
        'pretrained': {
                    'pretrained_path': None,
                    'pretrained_path_xfold': None,
                    'n_module_grow': None,
        },
        # Gradient checkpointing
        'use_checkpoint': True,
    }
)

def update_config(config):
    # TODO: update missing params
    for key in XFOLD_CONFIG:
        if key in config:
            if isinstance(config[key], dict):
                for key_ in XFOLD_CONFIG[key]:
                    if key_ not in config[key]:
                        config[key][key_] = XFOLD_CONFIG[key][key_]
        else:
            config[key] = XFOLD_CONFIG[key]

    return config

if __name__ == '__main__':
    print(XFOLD_CONFIG)
    Box(dict(XFOLD_CONFIG.to_dict())).to_yaml('../configs/model.yaml')
