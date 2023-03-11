# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder

from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOv5PoseKptCoder(BaseBBoxCoder):
    """YOLOv5 BBox coder.

    This decoder decodes pred bboxes (delta_x, delta_x, w, h) to bboxes (tl_x,
    tl_y, br_x, br_y).
    """

    def encode(self, **kwargs):
        """Encode deltas between bboxes and ground truth boxes."""
        pass

    def decode(self, priors: torch.Tensor, pred_kpts: torch.Tensor,
               stride: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode regression results (delta_x, delta_x, w, h) to bboxes (tl_x,
        tl_y, br_x, br_y).

        Args:
            priors (torch.Tensor): Basic boxes or points, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # pred_kpts = pred_kpts.sigmoid()
        bs, n, _ = pred_kpts.shape
        pred_kpts = pred_kpts.view(bs, n, -1, 2)

        x_center = (priors[..., 0] + priors[..., 2]) * 0.5
        y_center = (priors[..., 1] + priors[..., 3]) * 0.5

        stride = stride[None, :, None, None]
        pred_kpts = (pred_kpts - 0.5) * 2 * stride

        # The anchor of mmdet has been offset by 0.5
        x_center_pred = pred_kpts[..., 0] + x_center[..., None]
        y_center_pred = pred_kpts[..., 1] + y_center[..., None]

        return torch.cat(
            [x_center_pred.unsqueeze(-1),
             y_center_pred.unsqueeze(-1)], dim=-1)
