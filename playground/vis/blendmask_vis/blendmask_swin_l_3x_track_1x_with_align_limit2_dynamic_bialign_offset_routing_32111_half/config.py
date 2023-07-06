import os.path as osp

from cvpods.configs.blendmask_config import BlendMaskConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/mnt/lustre/share_data/lixiangtai/pretrained/coco/blendmask_swin_l_3x.pth",
        SWIN=dict(
            EMBED_DIM=192,
            OUT_FEATURES=["stage2", "stage3", "stage4", "stage5"],
            DEPTHS=[2, 2, 18, 2],
            NUM_HEADS=[6, 12, 24, 48],
            WINDOW_SIZE=7,
            MLP_RATIO=4,
            DROP_PATH_RATE=0.2,
            APE=False, ),
        FPN=dict(
            TOP_LEVELS=2,
            IN_FEATURES=["stage2", "stage3", "stage4", "stage5"]
        ),
        TRACK_ON=True,
        FCOS=dict(
            USE_DEFORMABLE=True,
            NUM_CLASSES=40,
        ),
        BLENDMASK=dict(
            TEMPORAL_ALIGN=dict(
                IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
                NUM_CONVS=2,
                ALIGN_CHANNELS=256,
                NUM_CONV_ROUTING_LIST=[3, 2, 1, 1, 1],
                USE_DCN_v2=True,
                USE_DCN=True,
                NORM="GN",
                NUM_OFFSET_GEN_CONV=2,
                SHARE_CONVS=True,
            ),
        ),
    ),
    DATASETS=dict(
        TRAIN=("youtubevislimit_train",),
        TEST=("youtubevis_valid",),
        REFERENCE_RANGE=2,
        ONLY_PREVIOUS=False
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(3840, 5280),
            MAX_ITER=5760,
            WARMUP_FACTOR=1.0 / 80,
            WARMUP_ITERS=1000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.005,
            WEIGHT_DECAY=0.0001,
            MOMENTUM=0.9,
        ),
        IMS_PER_BATCH=32,
        IMS_PER_DEVICE=4,
        CHECKPOINT_PERIOD=100000,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeLongestShortestEdgeVideo",
                 dict(img_scales=[(360, 649), (480, 960)],
                      sample_style="range",
                      keep_ratio=True)),
                ("RandomFlipVideo",
                 dict(prob=0.5,
                      horizontal=True,
                      vertical=False)),
            ],
            TEST_PIPELINES=[
                ("ResizeVideo",
                 dict(shape=(360, 640),
                      keep_ratio=True)),
            ],
        ),
        MASK_FORMAT="bitmask",
    ),
    TEST=dict(
        EVAL_PERIOD=-1,
    ),
    OUTPUT_DIR="./outputs",
)


class CustomFCOSConfig(BlendMaskConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
