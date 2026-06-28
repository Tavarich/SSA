# Implementation of Semantic and Sequential Alignment for Referring Video Object Segmentation

[[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Pan_Semantic_and_Sequential_Alignment_for_Referring_Video_Object_Segmentation_CVPR_2025_paper.pdf)]

## Installation

Please read [INSTALL.md](https://github.com/tavarich/SSA/blob/main/assets/INSTALL.md) for details.

## Data Preparation

This project uses MeViS, Ref-YouTube-VOS and Ref-DAVIS17 datasets.

Please read [DATA.md](https://github.com/tavarich/SSA/blob/main/assets/DATA.md) for details.

## Inference

### MeViS

```shell
python train_net_ssa.py \
    --config-file configs/mevis/ssa_clip_bs8.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_outputs]
```

We provide the model checkpoints here.

|                | model weights                                                | output files                                                 | J&F   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- |
| fine-tuning    | [model](https://drive.google.com/file/d/1_TVp9Mdt0one-lknRsPDROqL5ne4xz42/view?usp=sharing) | [outputs](https://drive.google.com/file/d/14GlsqeIxENi-q7jClegnCPUtPKWmztOw/view?usp=sharing) | 48.56 |
| joint training | [model](https://drive.google.com/file/d/1GmCQd80OIY4fSNOLEmXbihTqqjwuLz29/view?usp=sharing) | [outputs](https://drive.google.com/file/d/1aamYqNER9B02r_qDoCPE9pErLY2W6SPc/view?usp=sharing) | 49.37 |

The `val` split is evaluated through the online evaluation [server](), whereas the `valu` split can be evaluated locally with `tools/eval_mevis.py`

### Ref-Youtube-VOS

```shell
python train_net_ssa.py \
    --config-file configs/mevis/ssa_clip_bs8.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_outputs]
```

We provide the model checkpoints here.

|                 | model weights                                                | J&F  |
| --------------- | ------------------------------------------------------------ | ---- |
| Ref-YouTube-VOS | [model](https://drive.google.com/file/d/1--gAGOjmRxyeA7kNoX9dsk0og5gkF5k-/view?usp=sharing) | 64.3 |

For Ref-YouTube-VOS, submit the result to the online evaluation [server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).

### Ref-DAVIS17

```shell
python train_net_ssa.py \
    --config-file configs/mevis/ssa_clip_bs8.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_outputs]
```

The model checkpoints are directly adopted from those trained on Ref-YouTube-VOS.

|             | model weights                                                | J&F  |
| ----------- | ------------------------------------------------------------ | ---- |
| Ref-DAVIS17 | [model](https://drive.google.com/file/d/1--gAGOjmRxyeA7kNoX9dsk0og5gkF5k-/view?usp=sharing) | 67.3 |

This project saves prediction masks for each object separately, which is not fully compatible with the official DAVIS evaluation script. 

An alternative way is using `tools/eval_davis.py` (modified from ``eval_mevis.py``) to evaluate the predictions directly. The metrics may differ slightly from those obtained with the official evaluation protocol.

The official evaluation code from [ReferFormer](https://github.com/wjn922/ReferFormer/blob/main/docs/Ref-DAVIS17.md) can be used through a postprocess step to merge prediction masks:

```shell
# postprocess step
python tools/postprocess_davis.py  --mask_dir [path_to_outputs] --output_dir [path_to_saves]

# evaluation step 
python tools/eval_davis_official.py --results_path [path_to_saves]/anno_{0,1,2,3}
```

## Training

Following  the setup of MeViS,, we employ Mask2Former (CLIP vision encoder as backbone) and train it on COCO dataset as the initialized weights.  The checkpoints of the CLIP-based Mask2Former is available [here](https://drive.google.com/file/d/1cjzb9McZpo1fbFmpAv5FLZUSgJfPMlrW/view?usp=sharing). 

Before loading the checkpoint, initialize the CLIP model either with setting `open_clip.create_model_and_transforms()` or downloading weights from  [Hugging Face](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup) and placing them under `open_clip_model/`.

### MeViS

Training for MeViS, use

```shell
python train_net_ssa.py \
    --config-file configs/mevis/ssa_clip_bs8.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_outputs]
```

### Ref-Youtube-VOS

Training for Ref-Youtube-VOS, use 

```shell
# pre-training stage
python train_net_ssa.py \
    --config-file configs/coco/ssa_clip_bs8.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_outputs]

# fine-tuning stage
python train_net_ssa.py \
    --config-file configs/ytvos/ssa_clip_bs8.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_outputs]
```

The training settings can be modified from the config-file in``./config`` directory.  

## Acknowledgement

This project is based on [MeViS](https://github.com/henghuiding/MeViS), [FC-CLIP]() and [ReferFormer](https://github.com/wjn922/ReferFormer). Thanks for these great works.

