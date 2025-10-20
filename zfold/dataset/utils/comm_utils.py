"""Common utility functions."""

import os
import json
import random
import logging
import hashlib
from datetime import datetime

import yaml

def zfold_init(path=None, verb_levl=None):
    """Initialize the tFold-xxx framework.

    Args:
    * (optional) path: path to the configuration YAML file
    * (optional) verb_levl: verbose level for the <logging> module

    Returns:
    * config: dict of configurations
    """

    # load configurations from the YAML file
    if path is not None and os.path.exists(path):
        with open(path, 'r') as i_file:
            config = yaml.safe_load(i_file)
    else:
        config = {'misc': {'verb_levl': 'INFO'}}  # default configurations

    # over-ride configurations with the additional JSON file
    def _set_val(cfg, keys, val):
        if len(keys) == 1:
            cfg[keys[0]] = val
        else:
            _set_val(cfg[keys[0]], keys[1:], val)

    jzw_path_key = 'JIZHI_WORKSPACE_PATH'
    if jzw_path_key in os.environ:
        jsn_fpath = os.path.join(os.getenv(jzw_path_key), 'job_param.json')
        if os.path.exists(jsn_fpath):
            with open(jsn_fpath, 'r') as i_file:
                jsn_data = json.load(i_file)
            for key, val in jsn_data.items():
                _set_val(config, key.split('.'), val)

    # over-ride the verbose level configuration
    if verb_levl is not None:
        assert verb_levl in ['ERROR', 'WARNING', 'INFO', 'DEBUG'], \
            'unrecognized verbose level: ' + verb_levl
        config['misc']['verb_levl'] = verb_levl  # highest priority

    # configure the logging facility
    logging.basicConfig(
        format='[%(asctime)-15s %(levelname)s %(filename)s:L%(lineno)d] %(message)s',
        level=config['misc']['verb_levl'],
    )

    # display all the configuration items
    def _show_cfg(cfg, prefix=''):
        for key, val in cfg.items():
            if not isinstance(val, dict):
                logging.info('%s%s: %s' % (prefix, key, str(val)))
            else:
                logging.info('%s%s:' % (prefix, key))
                _show_cfg(val, prefix + '  ')

    logging.info('=== CONFIGURATION - BELOW ===')
    _show_cfg(config)
    logging.info('=== CONFIGURATION - ABOVE ===')

    return config


def get_md5sum(x_str, as_int=False):
    """Get the MD5 sum of the given string.

    Args:
    * x_str: input string
    * as_int: whether to return the MD5 sum as an integer

    Returns:
    * md5sum: MD5 sum string / integer
    """

    md5sum = hashlib.md5(x_str.encode('utf-8')).hexdigest()
    if as_int:
        md5sum = int(md5sum, 16)

    return md5sum


def get_rand_str():
    """Get a randomized string.

    Args: n/a

    Returns:
    * rand_str: randomized string
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    rand_val = random.random()
    rand_str_raw = '%s_%f' % (timestamp, rand_val)
    rand_str = hashlib.md5(rand_str_raw.encode('utf-8')).hexdigest()

    return rand_str


def get_num_threads():
    """Get the number of parallel threads.

    Arg: n/a

    Returns:
    * num_threads: number of parallel threads
    """

    if 'NUM_THREADS' not in os.environ:
        num_threads = 1
    else:
        num_threads = int(os.getenv('NUM_THREADS'))

    return num_threads


def make_config_list(**kwargs):
    """Make a list of configurations from (key, list of values) pairs.

    Args:
    * kwargs: (key, list of values) pairs

    Returns:
    * config_list: list of configurations
    """

    config_list = []
    for key, values in kwargs.items():
        if not config_list:
            config_list = [{key: value} for value in values]
        else:
            config_list_new = []
            for config in config_list:
                config_list_new.extend([{**config, key: value} for value in values])
            config_list = config_list_new

    return config_list
