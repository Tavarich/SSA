## Data Preparation

The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. 

```
$DETECTRON2_DATASETS/
├── mevis
├── ref-youtube-vos
├── ref-davis
└── coco
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. Alternatively, the datasets are expected to be placed under `./datasets`. 

### MeViS

Download MeViS dataset from [MeViS](https://github.com/henghuiding/MeViS) project.

```
mevis
├── train                       // Split Train
│   ├── JPEGImages
│   ├── mask_dict.json
│   └── meta_expressions.json
│
├── valid_u                     // Split Val^u
│   ├── JPEGImages
│   ├── mask_dict.json
│   └── meta_expressions.json
│
└── valid                       // Split Val
    ├── JPEGImages
    └── meta_expressions.json
```

### Ref-YouTube-VOS

Following [ReferFormer](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md), download the dataset from the competition's website [here](https://competitions.codalab.org/competitions/29139#participate-get_data). The directory structure are expected to be the following:

```
ref-youtube-vos
├── meta_expressions
├── train
│   ├── JPEGImages
│   ├── Annotations
│   ├── meta.json
└── valid
    └── JPEGImages
```

Note that the original `valid` expression JSON file contains videos from both the evaluation `valid` split and the `test` split. You need to filter out the `test` videos before using it.

### Ref-DAVIS17

Follow the data preparation instructions from [ReferFormer](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md) to download and process the Ref-DAVIS17 dataset. The processed directory structure are expected to  be as follows:

```shell
ref-davis
├── meta_expressions
├── train
│   ├── JPEGImages
│   ├── Annotations
│   └── meta.json
└── valid
    ├── JPEGImages
    ├── Annotations
    └── meta.json
```

### RefCOCO

Following [ReferFormer](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md),  download the dataset from the official website [COCO](https://cocodataset.org/#download). RefCOCO/+/g use the COCO2014 train split, and download the annotation files from [github](https://github.com/lichengunc/refer).

```shell
coco
├── train2014
├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
└── refcocog
    ├── instances.json
    ├── refs(google).p
    └── refs(unc).p
```
