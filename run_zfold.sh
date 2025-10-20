
ZFOLD_DPATH="/sugon_store/zhengliangzhen/P450/Z-Fold"
ZFOLD_BIN="${ZFOLD_DPATH}/envs/zFold/bin"
# m4
#MODEL_NAME="m4-384_256_lm4_lp4_md128_mp0.15_gr1_bs64_pld0.3-MSATrans"
# m2
MODEL_NAME="m2-384_256_lm4_lp4_md128_mp0.15_gr1_bs128_pld0.3-MSATrans"

if [ $# -lt 2 ]; then 
  echo "usage: run_zfold.sh input.a3m output/"
  exit 0;
fi 

msa=$1
out=$2
oa3m="$out/input.a3m"

mkdir -pv $out 
$ZFOLD_BIN/python $ZFOLD_DPATH/scripts/standardize_a3m.py $msa $oa3m
if [ `ls $out | grep "plddt" | wc -l` -eq 1 ]; then 
  echo "find previous output plddt file, exit now!!!"
  exit 0;
fi 

cmd="$ZFOLD_BIN/zfold_predict_e2e_single \
     --msa_paths $oa3m --tpl_paths=fake.pkl\
     --save_npzs $out/features2d.npz \
     --save_pdb_dirs $out --config_yaml \
     $ZFOLD_DPATH/zFold-ckp/$MODEL_NAME/model.yaml \
     --weight_path $ZFOLD_DPATH/zFold-ckp/$MODEL_NAME/checkpoint_slim.pt"

echo "Running tasks with zFold: "
echo $cmd 

$cmd 
wait 

echo "Complete structure prediction with zFold!!!"
exit 0;