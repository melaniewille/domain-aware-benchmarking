# Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality

This is the official repository for the paper: **"Why Domain Matters: Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality"**, accepted at WACV 2027.

The preprint is available on [arXiv](https://arxiv.org/abs/2607.10575). 
We will release a corresponding video on [YouTube](https://www.youtube.com/@melaniewille) soon!

Our work contributes to the field of underwater object detection by providing a systematic analysis of how different underwater conditions influence both human annotation quality and object detector performance. While previous works mainly rely on aggregate performance metrics or synthetic domain variations, we introduce a framework that characterizes underwater images based on appearance, scene composition, and acquisition geometry. Using this framework, we demonstrate that both annotation quality and detection performance vary substantially across domains, revealing challenges that are hidden by conventional evaluation approaches. Importantly, conditions that challenge human annotators do not necessarily challenge object detectors in the same way. By making domain shift measurable and interpretable, our work provides a foundation for more informative benchmarking, targeted data collection and annotation, and the development of more robust underwater object detectors for reliable marine environmental monitoring.


This repository provides the code and instructions to reproduce our analysis of human annotation quality using the RUOD and RUOD-R datasets, as well as our domain-aware benchmarking of YOLO26n, Faster R-CNN, and RT-DETR on DUO, RUOD-R, and UTDAC. The experiments rely on domain labels assigned using our [Underwater Domain Labeling Framework](https://github.com/melaniewille/underwater_domain_labeling).

If this repository and/or our paper contributes to your research, please consider citing the publication below.


### Bibtex
```
@article{wille2026domain,
  title={Why Domain Matters: Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality},
  author={Wille, Melanie and Miller, Dimity and Fischer, Tobias and Raine, Scarlett},
  journal={arXiv preprint arXiv:2607.10575},
  year={2026}
}
```

## Table of Contents

- [Repo Usage](#repo-usage)  
- [Datasets](#datasets)  
- [Annotation Study](#annotation-study)  
- [Detection Study](#detection-study)  
- [Acknowledgements](#acknowledgements)

## Repo Usage
This repository contains the code used to reproduce the experiments presented in our WACV 2027 paper. The experiments are divided into two main components:

**1. Annotation Study**, where we investigated the annotation differences between the original RUOD dataset and its revised version (RUOD-R) across underwater domains

**2. Detection Study**, where we trained YOLO26n, Faster R-CNN and RT-DETR on a combined underwater object detection dataset and tested performance across underwater domains

Both studies use image-level domain labels generated using our [Underwater Domain Labeling Framework](https://github.com/melaniewille/underwater_domain_labeling).

The following sections describe the required datasets, experimental procedures and corresponding scripts.



## Datasets
All experiments focus on four marine object classes: echinus, holothurian, scallop, and starfish. The datasets used in our experiments are available on [Hugging Face](https://huggingface.co/datasets/melaniewille/domain-aware-benchmarking-uod). 

For the annotation study, we compare the original RUOD annotations with the revised RUOD-R annotations using only images shared between both datasets after filtering for our four target classes. The shared images can be dowloaded from [Hugging Face](https://huggingface.co/datasets/melaniewille/domain-aware-benchmarking-uod), while their corresponding annotations from each dataset version can be found under [datasets/comparison-instances/](./annotation-study/datasets/comparison-instances/). The domain labels assigned to the shared images using our [Underwater Domain Labeling Framework](https://github.com/melaniewille/underwater_domain_labeling) are provided in [domain-labels/](./annotation-study/domain-labels/). They include both versions (based on RUOD and RUOD-R) for completeness. However, our analysis uses the RUOD-R domain labels, as the revised annotations are assumed to more accurately represent the actual scene composition.

```
.
├── annotation-study/
│   ├── datasets/
│   │   ├── comparison-instances/
│   │   │   ├── comparison-ruod-instances.json
│   │   │   └── comparison-ruod-r-instances.json
│   │   └── shared-images/
│   │       └── [on hugging face]
│   ├── domain-labels/
│   │       ├── comparison_domain_labels_ruod.json
│   │       └── comparison_domain_labels_ruod-r.json
│   └── ...
└── ...
```

For the detection study, we combine DUO, RUOD-R and UTDAC into a single dataset with our four target classes. We generate five independent random training, validation, and testing splits using a 70/10/20 ratio. All detectors are trained and evaluated using the same splits to ensure a fair comparison. The test data is further organized according to the image-level domain properties assigned by our [Underwater Domain Labeling Framework](https://github.com/melaniewille/underwater_domain_labeling), with the corresponding subsets defined through YAML and TXT files. An overview of the domain labels can also be dowloaded with the images and annotations from [Hugging Face](https://huggingface.co/datasets/melaniewille/domain-aware-benchmarking-uod).
```
.
├── detection-study/
│   ├── datasets/
│   │   └── detector-analysis/
│   │       └── [on hugging face]
│   ├── yamls-detector-analysis/
│   │       ├── all_dark_data_test.yaml
│   │       ├── all_bright_data_test.yaml
│   │       ├── ...
│   │       ├── v2_all_dark_data_test.yaml
│   │       └── ...
│   └── ...
└── ...
```

## Annotation Study
To reproduce our annotation study and compare the original RUOD annotations against the revised RUOD-R annotations across the eight domain categories introduced in our framework, please follow the steps below:

**Step 1 — Per-image annotation comparison**

_Script_: [01-reannotation-per-image.py](./annotation-study/01-reannotation-per-image.py)

This script compares the original RUOD and revised RUOD-R ground-truth annotations for each shared image (kept / added / removed). Bounding boxes are matched using an Intersection over Union (IoU) threshold of 0.5 by default.

**Input**:
- Original RUOD annotations
- Revised RUOD-R annotations
- Image-level domain labels

**Output**:
- results/reannotation-per-image-with-domain.csv


**Step 2 — Domain-dependent correction rates**

_Script_: [02-correction-rate-evaluation.py](./annotation-study/02-correction-rate-evaluation.py)

This script calculates and summarizes annotation correction rates across the different underwater domain categories.

**Input**:
- results/reannotation-per-image-with-domain.csv

**Output**:
- results/correction-rates-per-domain.csv


**Step 3 — Statistical significance and robustness of results**

_Script_: [03-statistical-analysis.py](./annotation-study/03-statistical-analysis.py)

This script evaluates the statistical significance and robustness of domain-dependent differences in annotation correction rates. It applies Kruskal–Wallis tests to identify differences across domain categories, Spearman's rank correlation to assess ordinal relationships, and pairwise Mann–Whitney U tests to compare individual categories. Bootstrap resampling is used to assess the stability of the statistical results.

**Input**:
- results/reannotation-per-image-with-domain.csv

**Output**:
- results/kruskal.csv
- results/spearman.csv
- reults/pairwise-mannwhitney.csv
- results/bootstrap-kruskal-spearman.csv
- results/bootstrap-mannwhitney.csv




## Detection Study
This repository provides the training and testing scripts for [YOLO26n](./detection-study/yolo), [Faster RCNN](./detection-study/faster-rcnn), and [RT-DETR](./detection-study/rt-detr) object detection models. 


```
.
├── detection-study/
│   ├── yolo/
│   │   ├── train_yolo_mixed.py
│   │   ├── detector_ablation_yolo_visibility_test.py
│   │   ├── detector_ablation_yolo_illumination_test.py
│   │   ├── ...
│   │   ├── v2_train_yolo_mixed.py
│   │   ├── v2_detector_ablation_yolo_visibility_test.py
│   │   ├── v2_detector_ablation_yolo_illumination_test.py
│   │   └── ...
│   └── faster-rcnn/
│   │   ├── train_fasterrcnn.py
│   │   ├── test_fasterrcnn.py
│   │   ├── v2_train_fasterrcnn.py
│   │   ├── v2_test_fasterrcnn.py
│   │   └── ...
│   └── rt-detr/
│   │   ├── train_rtdetr_mixed.py
│   │   ├── detector_ablation_rtdetr_visibility_test.py
│   │   ├── detector_ablation_rtdetr_illumination_test.py
│   │   ├── ...
│   │   ├── v2_train_rtdetr_mixed.py
│   │   ├── v2_detector_ablation_rtdetr_visibility_test.py
│   │   ├── v2_detector_ablation_rtdetr_illumination_test.py
│   │   └── ...
│   └── yamls-detector-analysis/
│   │   └── ...
│   └── datasets/
│       └── ...
└── ...
```

The **results** reported in our **main paper** are the average over 5 independent training and testing runs.



## Acknowledgements
This work was supported by the QUT Centre for Robotics, QUT Digital Research Infrastructure team for
HPC, and an ARC DECRA Fellowship DE240100149 to TF.

We also thank the contributors of the **DUO** dataset - check out their [Github](https://github.com/chongweiliu/DUO) and paper:
```
@inproceedings{liu2021dataset,
  title={A dataset and benchmark of underwater object detection for robot picking},
  author={Liu, Chongwei and Li, Haojie and Wang, Shuchang and Zhu, Ming and Wang, Dong and Fan, Xin and Wang, Zhihui},
  booktitle={2021 IEEE international conference on multimedia \& expo workshops (ICMEW)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
and the contributors of the **RUOD** dataset - check out their [Github](https://github.com/dlut-dimt/RUOD) and paper:
```
@article{fu2023rethinking,
  title={Rethinking general underwater object detection: Datasets, challenges, and solutions},
  author={Fu, Chenping and Liu, Risheng and Fan, Xin and Chen, Puyang and Fu, Hao and Yuan, Wanqi and Zhu, Ming and Luo, Zhongxuan},
  journal={Neurocomputing},
  volume={517},
  pages={243--256},
  year={2023},
  publisher={Elsevier}
}
```

We redistribute our derived versions of the datasets according to:

- a **private** permission for the **DUO** dataset from the original authors (email correspondence, October 2025). Further redistribution or commercial use may require additional consent from Liu et al.
- the **Apache License 2.0** for the **RUOD** dataset, under which it was originally released. We release our subset RUOD-4C under the same license.




