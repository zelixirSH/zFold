import shutil
import torch
from box import Box

def load_pretrain(model,
                  pretrain_file,
                  is_GPU = False,
                  is_fair = False,
                  is_origin = False,
                  skip_keys = None,
                  from_xfold_to_xfold2d = False):

    print('    loading', pretrain_file)

    pretrain_state_dict = torch.load(pretrain_file) if is_GPU else torch.load(pretrain_file, map_location='cpu')

    if is_fair:
        pretrain_state_dict = pretrain_state_dict['model']

    elif is_origin:
        pretrain_state_dict = pretrain_state_dict['model_state_dict']

    state_dict = model.state_dict()

    for key in state_dict.keys():

        skip = False
        if isinstance(skip_keys, list):
            for sk in skip_keys:
                if sk in key:
                    skip = True
        if skip:
            print(f'skip loading: {key}')
            continue

        new_key = key

        if from_xfold_to_xfold2d:
            new_key = 'xfold2d.' + new_key

        if new_key in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict[new_key]

        elif ('encoder.xfold.' + new_key) in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict['encoder.xfold.' + new_key]

        elif ('encoder.model.' + new_key) in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict['encoder.model.' + new_key]

        elif ('model.' + new_key) in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict['model.' + new_key]

        else:
            print(f'{new_key} not in pretrain_state_dicts')

    model.load_state_dict(state_dict)

    return model

