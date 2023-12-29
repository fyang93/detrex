import itertools

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

# register_coco_instances("my_dataset_train", {}, '/path/to/train.json', '/path/to/train/images')
# register_coco_instances("my_dataset_test", {}, '/path/to/test.json', '/path/to/test/images')

### train
register_coco_instances("caltech_pedestrians_train", {}, '../../data/smoke-fire-person-dataset/person/caltech_pedestrians/coco/train.json', '../../data/smoke-fire-person-dataset/person/caltech_pedestrians/yolo/train/images')
register_coco_instances("crowd_human_train", {}, '../../data/smoke-fire-person-dataset/person/crowd_human/coco/train.json', '../../data/smoke-fire-person-dataset/person/crowd_human/train/images')
register_coco_instances("visdrone_train", {}, '../../data/smoke-fire-person-dataset/person/VisDrone/train/train.json', '../../data/smoke-fire-person-dataset/person/VisDrone/train/images')
register_coco_instances("visdrone_val", {}, '../../data/smoke-fire-person-dataset/person/VisDrone/val/val.json', '../../data/smoke-fire-person-dataset/person/VisDrone/val/images')

### test
# register_coco_instances("caltech_pedestrians_test", {}, '/data/smoke-fire-person-dataset/person/caltech_pedestrians/coco/test.json', '/data/smoke-fire-person-dataset/person/caltech_pedestrians/yolo/test/images')
# register_coco_instances("crowd_human_val", {}, '/data/smoke-fire-person-dataset/person/crowd_human/coco/val.json', '/data/smoke-fire-person-dataset/person/crowd_human/val/images')
register_coco_instances("visdrone_test", {}, '../../data/smoke-fire-person-dataset/person/VisDrone/test/test.json', '../../data/smoke-fire-person-dataset/person/VisDrone/test/images')
# register_coco_instances("merged", {}, '../../data/smoke-fire-person-dataset/person/merged_test/results/merged/annotations/merged.json', '../../data/smoke-fire-person-dataset/person/merged_test/results/merged/images')


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="visdrone_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="visdrone_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
