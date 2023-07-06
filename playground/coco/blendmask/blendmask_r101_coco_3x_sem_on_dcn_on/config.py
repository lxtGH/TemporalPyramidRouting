import os.path as osp

from cvpods.configs.blendmask_config import BlendMaskConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/mnt/lustre/share_data/lixiangtai/pretrained/R-101.pkl",
        RESNETS=dict(
            DEPTH=101,
            DEFORM_ON_PER_STAGE=[False, False, True, True],
            DEFORM_MODULATED=True,
        ),
        BASIS_MODULE=dict(
            NAME="ProtoNet",
            NUM_BASES=4,
            LOSS_ON=False,
            CONDTION_INST_LOSS_ON=True,
            ANN_SET="coco",
            CONVS_DIM=128,
            IN_FEATURES=["p3", "p4", "p5"],
            NORM="SyncBN",
            NUM_CONVS=3,
            COMMON_STRIDE=8,
            NUM_CLASSES=80,
            LOSS_WEIGHT=0.3
        ),
        FCOS=dict(
            USE_DEFORMABLE=True,
        )
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(210000, 250000),
            MAX_ITER=270000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=10000,
    ),
    OUTPUT_DIR=osp.join(
        './outputs/blendMask_r50',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class CustomFCOSConfig(BlendMaskConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
