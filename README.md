l![header](images/header_new.jpeg)

# Z-Fold
State of the art 1D & 2D modules for Protein Structure Prediction, complete with a set of promising experimental features from recent papers (tFold-DistNet, RoseTTAFold, and Alphafold2).

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.9.0
* Python version >= 3.7

```bash
$ cd Z-Fold
$ pip install .
```

* The models (checkpoints) could be accessed through the following link:
```
https://gofile.me/7zo5Q/QHXoVKyR1
```


## Usage

Example: to run a trosetta-npz & pdb prediction of xfold_e2e model with command line

```bash
CKPT_DIR=/share/taosheng/code/zfold_ckpts/m4-384_256_lm4_lp4_md128_mp0.15_gr1_bs64_pld0.3-MSATrans
EXAMPLE_DIR=/share/taosheng/code/deploy/zfold_example
BIN_DIR=/share/taosheng/anaconda3/bin

$BIN_DIR/zfold_predict_e2e_single --msa_paths $EXAMPLE_DIR/T1024-D1.a3m --tpl_paths $EXAMPLE_DIR/T1024-D1.pkl --save_npzs ./tmp/tmp.npz --save_pdb_dirs ./tmp \
                    --config_yaml $CKPT_DIR/model.yaml \
                    --weight_path $CKPT_DIR/checkpoint_slim.pt \
                    --is_gpu True --is_fp16 True

```

## One stage End-to-end training

* m: n_module        # number of 2-track modules
* bs: batch size
* pld: progressive layer drop rate of XFold
* gr: global recycling  # default: 1, this means no global recycling
* md: msa depth of training # default: 128
* mp: msa mask prob     # default: 0.15

### End-to-end training Using hybrid dataset (RCSB-PDB-28k + trrosetta)

|  |casp14 - tm score                       | gdt_ts | lddt_ca | lddt   | contact | cameo5m - tm_scr | gdt_ts | lddt_ca | lddt   | contact                    |
|-----------------------------------------|--------|---------|--------|---------|------------------|--------|---------|--------|---------|---------------------|
| AlphaFold2 (model1)                     | 0.8823 | 0.8562  | 0.8677 | 0.7989  | 0.7871           | 0.8989 | 0.8782  | 0.9135 | 0.8544  | 0.8381(5 model ave) |
||||
| m1 bs64 pld0.1 gr1 md128 mp0.15| 0.8452 | 0.8071  | 0.8294 | 0.7572  | 0.764            | 0.8534 | 0.8162  | 0.8614 | 0.7937  | 0.8148              |
| m1 bs16 pld0.3 gr2 md256 mp0.4 | 0.837  | 0.792   | 0.8156 | 0.7417  |                  | 0.8545 | 0.8109 | 0.8522  | 0.781  | 0.8207               |
| m1 bs64 pld0.3 gr2 md256 mp0.4 | 0.8467 | 0.8053  | 0.8305 | 0.7565  | 0.7605           | 0.8617 | 0.8217  | 0.8672 | 0.7974  | 0.8202              |
| m2 bs16 pld0.3 gr2 md256 mp0.4 | 0.844  | 0.8036  | 0.8234 | 0.7501  | 0.7613           | 0.8547 | 0.812   | 0.8551 | 0.785   | 0.8152              |

### End-to-end training with self-distillation (RCSB-PDB-28k + trrosetta + 360k AF2 Deocys)

|  |  casp14 - tm score                 | casp14 - gdt_ts | casp14 - lddt_ca | casp14 - lddt   | casp14 - contact | cameo5m - tm_scr |  cameo5m - gdt_ts |  cameo5m - lddt_ca |  cameo5m - lddt   | cameo5m - contact |                        |
|-----------------------------------------|--------|---------|--------|---------|------------------|--------|---------|--------|---------|---------------------|---|
| AlphaFold2 (model1)                     | 0.8823 | 0.8562  | 0.8677 | 0.7989  | 0.7871           | 0.8989 | 0.8782  | 0.9135 | 0.8544  | 0.8381(5 model ave) |   |
|                                         |        |         |        |         |                  |        |         |        |         |                     |   |
| m1 bs16 pld0.1 gr1                      | 0.8312 | 0.7886  | 0.8155 | 0.7434  | 0.7527           | 0.8535 | 0.8082  | 0.8531 | 0.7858  | 0.8141              |   |
| m1 bs16 pld0.3 gr1                      | 0.8298 | 0.7859  | 0.8124 | 0.7374  | 0.754            | 0.8532 | 0.8114  | 0.854  | 0.7819  | 0.8112              |   |
| m1 bs64 pld0.3 gr1                      | 0.8626 | 0.8272  | 0.8479 | 0.7736  | 0.7785           | 0.8714 | 0.8354  | 0.8779 | 0.8096  | 0.8242              |   |
| m1 bs128 pld0.3 gr1                     | 0.8675 | 0.8357  | 0.8538 | 0.7808  | 0.7811           | 0.8786 | 0.847   | 0.8879 | 0.8212  | 0.8331              |   |
| m1 bs64 pld0.3 gr2                      | 0.8713 | 0.8376  | 0.8569 | 0.7832  | 0.7795           | 0.8781 | 0.8431  | 0.8861 | 0.8192  | 0.832               |   |
| m1 bs64 pld0.3 gr2 md256 mp0.4          | 0.8712 | 0.8402  | 0.8573 | 0.784   | 0.7878           | 0.8752 | 0.8407  | 0.884  | 0.8175  | 0.8309              |   |
|                                         |        |         |        |         |                  |        |         |        |         |                     |   |
| m2 bs16 pld0.1 gr1                      | 0.8549 | 0.8159  | 0.8367 | 0.7653  | 0.7726           | 0.8596 | 0.8147  | 0.8636 | 0.7974  | 0.8206              |   |
| m2 bs16 pld0.3 gr1                      | 0.8572 | 0.8195  | 0.8405 | 0.7667  | 0.7693           | 0.8695 | 0.8312  | 0.8743 | 0.8048  | 0.8254              |   |
| m2 bs64 pld0.3 gr1                      | 0.8782 | 0.8455  | 0.8635 | 0.7912  | 0.7917           | 0.8836 | 0.8523  | 0.8936 | 0.828   | 0.8373              |   |
| m2 bs128 pld0.3 gr1                     |  0.8835|0.8555   | 0.8717 |0.7997   | 0.7926           | 0.8852 |0.8575   |0.8982  |0.8342   |0.8388 ||
| m2 bs64 pld0.3 gr2 md256 mp0.4          | 0.8842 | 0.8547  | 0.8696 | 0.7973  | 0.7954           | 0.8839 | 0.8533  | 0.895  | 0.8287  | 0.8375              |   |
| m2 bs64 pld0.3 gr2 md256 mp0.4 (32A100) | 0.881  | 0.8508  | 0.8673 | 0.7955  | 0.7972           | 0.8796 | 0.8479  | 0.8902 | 0.825   | 0.8337              |   |
|                                         |        |         |        |         |                  |        |         |        |         |                     |   |
| m4 bs16 pld0.1 gr1                      | 0.8579 | 0.8197  | 0.8388 | 0.7678  | 0.7731           | 0.8597 | 0.8192  | 0.8622 | 0.7965  | 0.8207              |   |
| m4 bs16 pld0.3 gr1                      | 0.8592 | 0.823   | 0.8429 | 0.7694  | 0.772            | 0.8729 | 0.836   | 0.8778 | 0.8076  | 0.8262              |   |
| m4 bs64 pld0.3 gr1                      | 0.8792 | 0.8518  | 0.8676 | 0.7961  | 0.7957           | 0.8851 | 0.8559  | 0.8955 | 0.8307  | 0.8358              |   |

### 2D-Prediction-Only training with self-distillation (RCSB-PDB-28k + trrosetta + 360k AF2 Deocys)
|  |casp14 - contact                             | cameo5m - contact                      |
|-------------------------------------|---------|---------------------|
| AlphaFold2 (model1)                 | 0.7871  | 0.8381(5 model ave) |
| m1 bs16 pld0.1 gr1                  | 0.765   | 0.82                |
| m1 bs16 pld0.1 gr1 + ESM1B          | 0.7644  | 0.8245              |
| m1 bs64 pld0.1 gr1                  | 0.7948  | 0.8349              |
| m2 bs16 pld0.1 gr1                  | 0.7875  | 0.8318              |
| m2 bs64 pld0.1 gr1                  | 0.8004  | 0.8376              |
| m4 bs16 pld0.1 gr1                  | 0.7915  | 0.8303              |

All models above are evaluated with the same input MSAs (bfd_metaclust.a3m.files.3.1e-0 for CASP14 targets and UniRef90 msa for CAMEO5M).

All models are trained 125k iters with a learning rate of 0.0003. Better performance can be achieved by scaling up the crop_size, datasets and model depth.

## Citations

```bibtex
@article {Baek2021.06.14.448402,
    author  = {Baek, Minkyung and DiMaio, Frank and Anishchenko, Ivan and Dauparas, Justas and Ovchinnikov, Sergey and Lee, Gyu Rie and Wang, Jue and Cong, Qian and Kinch, Lisa N. and Schaeffer, R. Dustin and Mill{\'a}n, Claudia and Park, Hahnbeom and Adams, Carson and Glassman, Caleb R. and DeGiovanni, Andy and Pereira, Jose H. and Rodrigues, Andria V. and van Dijk, Alberdina A. and Ebrecht, Ana C. and Opperman, Diederik J. and Sagmeister, Theo and Buhlheller, Christoph and Pavkov-Keller, Tea and Rathinaswamy, Manoj K and Dalwadi, Udit and Yip, Calvin K and Burke, John E and Garcia, K. Christopher and Grishin, Nick V. and Adams, Paul D. and Read, Randy J. and Baker, David},
    title   = {Accurate prediction of protein structures and interactions using a 3-track network},
    year    = {2021},
    doi     = {10.1101/2021.06.14.448402},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/06/15/2021.06.14.448402},
    eprint  = {https://www.biorxiv.org/content/early/2021/06/15/2021.06.14.448402.full.pdf},
    journal = {bioRxiv}
}
```

```bibtex
@misc{unpublished2021alphafold2,
    title   = {Alphafold2},
    author  = {John Jumper},
    year    = {2020},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.BM}
}
```
