# Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality

This is the official repository for the paper: **"Why Domain Matters: Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality"**, accepted at WACV 2027.

The preprint is available on [arXiv](https://arxiv.org/abs/2607.10575). 
We will release a corresponding video on [YouTube](https://www.youtube.com/@melaniewille) soon!

Our work contributes to the field of underwater object detection by providing a systematic analysis of how different underwater conditions influence both human annotation quality and object detector performance. While previous works mainly rely on aggregate performance metrics or synthetic domain variations, we introduce a framework that characterizes underwater images based on appearance, scene composition, and acquisition geometry. Using this framework, we demonstrate that both annotation quality and detection performance vary substantially across domains, revealing challenges that are hidden by conventional evaluation approaches. Importantly, conditions that challenge human annotators do not necessarily challenge object detectors in the same way. By making domain shift measurable and interpretable, our work provides a foundation for more informative benchmarking, targeted data collection and annotation, and the development of more robust underwater object detectors for reliable marine environmental monitoring.

The experiments rely on domain labels assigned using our [Underwater Domain Labeling Framework](https://github.com/melaniewille/underwater_domain_labeling).

This repository provides the code and instructions to reproduce our analysis of human annotation quality using the RUOD and RUOD-R datasets, as well as our domain-aware benchmarking of YOLO26n, Faster R-CNN, and RT-DETR on DUO, RUOD-R, and UTDAC.

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
soon


## Datasets
The data we used is available on [Hugging Face]([https://huggingface.co/](https://huggingface.co/datasets/melaniewille/domain-aware-benchmarking-uod)).  

You can find the shared 


## Annotation Study
soon

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




