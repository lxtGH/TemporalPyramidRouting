import os.path as osp

from cvpods.configs.blendmask_config import BlendMaskConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/mnt/lustre/lixiangtai/pretrained/swin/swin_tiny_patch4_window7_224_d2.pth",
        BACKBONE=dict(FREEZE_AT=-1),
        SWIN=dict(
           EMBED_DIM=96,
           OUT_FEATURES=["stage2", "stage3", "stage4", "stage5"],
           DEPTHS=[2, 2, 6, 2],
           NUM_HEADS=[3, 6, 12, 24],
           WINDOW_SIZE=7,
           MLP_RATIO=4,
           DROP_PATH_RATE=0.2,
           APE=False,
        ),
        FPN=dict(
            TOP_LEVELS=2,
            IN_FEATURES=["stage2", "stage3", "stage4", "stage5"]
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
    ),

    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(210000, 250000),
            MAX_ITER=270000,
            WARMUP_FACTOR=0.01,
            WARMUP_ITERS=1000
        ),
        OPTIMIZER=dict(
            NAME="FullModelAdamWBuilder",
            BASE_LR=0.0001,
            BASE_LR_RATIO_BACKBONE=1.,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            CLIP_TYPE="norm",
            CLIP_VALUE=1.0,
            ENABLED=True,
            FULL_MODEL=True,
            NORM_TYPE=2.0,
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
        ),
        FORMAT="RGB"
    ),
    TEST=dict(
        EVAL_PERIOD=1000000,
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
