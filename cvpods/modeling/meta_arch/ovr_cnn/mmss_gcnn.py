"""
Implements the Multimedia Self-Supervised Grid-based (proposal-free) CNN framework
"""
import numpy as np
import torch
from torch import nn
from cvpods.structures import ImageList


class MMSSGridModel(nn.Module):
    """
    MMSS stands for multimedia self-supervised
    Main Model for pretraining
    """

    def __init__(self, cfg):
        super(MMSSGridModel, self).__init__()

        self.backbone = cfg.build_backbone(cfg)
        self.language_backbone = cfg.build_language_backbone(cfg)
        self.mmss_heads = cfg.build_mmss_heads(cfg,
            v_dim=self.backbone._out_feature_channels["res5"],
            l_dim=self.language_backbone.out_channels,
            loc_dim=2,
            backbone=self.language_backbone.body,
        )

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.mvm = cfg.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_VISUAL_MODELING
        self.spatial_dropout = cfg.MODEL.MMSS_HEAD.SPATIAL_DROPOUT

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        
    def forward(self, batched_inputs):
        """
        Arguments:
            batched_inputs["image"] (list[Tensor] or ImageList): images to be processed
            batched_inputs["annotation"] (list[str]): ground-truth captions for images (optional)

        Returns:
            result tuple: (dict[Tensor], dict[Tensor]): losses and other information.

        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        targets = [x["annotation"] for x in batched_inputs]

        visual_grid_features = self.backbone(images.tensor)["res5"]

        _, _, image_h, image_w = images.tensor.shape
        batch_size, dim, grid_h, grid_w = visual_grid_features.shape
        max_num_regions = grid_h * grid_w

        flattened_features = visual_grid_features.reshape(
            [batch_size, dim, max_num_regions]).permute(0, 2, 1)  # (b,c, h*w) -> (b, h*w, c)

        image_sizes = np.asarray(images.image_sizes, dtype=np.float32)
        grid_sizes = np.zeros(image_sizes.shape, dtype=np.int32)
        grid_sizes[:, 0] = np.ceil(image_sizes[:, 0] * grid_h / image_h)
        grid_sizes[:, 1] = np.ceil(image_sizes[:, 1] * grid_w / image_w)
        grid_mask = np.zeros([batch_size, grid_h, grid_w], dtype=np.uint8)
        for i in range(batch_size):
            grid_mask[i, :grid_sizes[i, 0], :grid_sizes[i, 1]] = 1
        flattened_mask = grid_mask.reshape([batch_size, max_num_regions]) # (b, h*w)

        loc_x = np.zeros([batch_size, grid_h, grid_w], dtype=np.float32)
        loc_y = np.zeros([batch_size, grid_h, grid_w], dtype=np.float32)
        for i in range(batch_size):
            y = (np.arange(grid_sizes[i, 0], dtype=np.float32) + 0.5) / grid_sizes[i, 0]
            x = (np.arange(grid_sizes[i, 1], dtype=np.float32) + 0.5) / grid_sizes[i, 1]
            loc_x[i, :grid_sizes[i, 0], :grid_sizes[i, 1]] = x[None, :]
            loc_y[i, :grid_sizes[i, 0], :grid_sizes[i, 1]] = y[:, None]
        flattened_loc = np.stack([loc_x, loc_y], axis=-1).reshape(
            [batch_size, max_num_regions, 2])
        flattened_loc = torch.tensor(flattened_loc).cuda() # (b, h*w, 2)

        if self.spatial_dropout > 0 and self.training:
            subsampled_features = []
            subsampled_loc = []
            new_mask = np.zeros([batch_size, self.spatial_dropout], dtype=np.uint8)
            for i in range(batch_size):
                idx = np.where(flattened_mask[i])[0]
                np.random.shuffle(idx)
                n = min(self.spatial_dropout, idx.shape[0])
                idx = idx[:n]
                subsampled_features.append(flattened_features[i, idx])
                subsampled_loc.append(flattened_loc[i, idx])
                new_mask[i, :n] = 1
            flattened_features = torch.nn.utils.rnn.pad_sequence(
                subsampled_features, batch_first=True)
            flattened_loc = torch.nn.utils.rnn.pad_sequence(
                subsampled_loc, batch_first=True)
            flattened_mask = new_mask

        input_image = {
            'region_features': flattened_features,
            'region_mask': torch.tensor(flattened_mask).cuda(),
            'region_loc': flattened_loc,
            'mvm_mask': torch.zeros(batch_size, max_num_regions).cuda(),
            'target_region_features': flattened_features,
        }
        if self.mvm:
            raise NotImplementedError

        input_caption = self.language_backbone(targets)

        mmss_outputs = {}
        mmss_losses = {}
        for head in self.mmss_heads:
            o, l = self.mmss_heads[head](input_image, input_caption)
            mmss_outputs.update(o)
            mmss_losses.update(l)

        # for v in mmss_losses.values():
        #     if torch.isnan(v):
        #         print(self.mmss_heads['GroundingHead'].log_info)
        #         print(image_sizes, grid_sizes)
        #         raise ValueError()
        if self.training:
            return mmss_losses
        else:
            return mmss_outputs