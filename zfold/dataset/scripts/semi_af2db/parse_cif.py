import os
import numpy as np
from gemmi import cif
import pickle
from p_tqdm import p_map

def parse_cif_file(cif_file):
    doc = cif.read_file(cif_file)
    block = doc.sole_block()
    label_seq_id = np.asarray(list(block.find_loop('_ma_qa_metric_local.label_seq_id')))
    label_seq_id = np.asarray([int(v) for v in label_seq_id])
    value = list(block.find_loop("_ma_qa_metric_local.metric_value"))
    value = np.asarray([float(v) for v in value])
    return os.path.basename(cif_file).replace('-model_v2.cif',''), label_seq_id, value

if __name__ == '__main__':
    pkl = '/mnt/superCeph2/private/user/shentao/database/af2db_v2/tmp.pkl'
    n_threads = 128

    cif_dir = '/mnt/superCeph2/private/user/shentao/database/af2db_v2/swissprot_cif_v2'
    args_list = os.listdir(cif_dir)
    args_list = [f'{cif_dir}/{v}' for v in args_list if v.endswith('.cif')]

    results = p_map(parse_cif_file, args_list, num_cpus=0.75)

    data = {}
    for re in results:
        data[re[0]] = (re[1], re[2])

    # save pkl
    with open(pkl, "wb") as fp:   #Pickling
        pickle.dump(data, fp, protocol = pickle.HIGHEST_PROTOCOL)

    # load pkl
    with open(pkl, "rb") as fp:   #Pickling
        data = pickle.load(fp)

