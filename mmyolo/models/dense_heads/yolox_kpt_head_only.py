# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import multi_apply, filter_scores_and_topk
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor
from mmengine.config import ConfigDict

from mmyolo.registry import MODELS, TASK_UTILS
from .yolov5_head import YOLOv5Head
from mmyolo.datasets.utils import Keypoints


@MODELS.register_module()
class YOLOXKptOnlyHeadModule(BaseModule):
    """YOLOXHead head module used in `YOLOX.

    `<https://arxiv.org/abs/2107.08430>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid
        stacked_convs (int): Number of stacking convs of the head.
            Defaults to 2.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Defaults to False.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        kpt_stacked_convs: int = 4,
        featmap_strides: Sequence[int] = [8, 16, 32],
        use_depthwise: bool = False,
        dcn_on_last_conv: bool = False,
        conv_bias: Union[bool, str] = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        self.kpt_stacked_convs = kpt_stacked_convs
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.num_base_priors = num_base_priors

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self.multi_level_kpt_convs = nn.ModuleList()
        # kpt related layers
        self.multi_level_conv_kpt = nn.ModuleList()
        self.multi_level_conv_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.multi_level_kpt_convs.append(self._build_kpt_stacked_convs())
            conv_kpt, conv_vis = self._build_predictor()
            self.multi_level_conv_kpt.append(conv_kpt)
            self.multi_level_conv_vis.append(conv_vis)

    def _build_kpt_stacked_convs(self) -> nn.Sequential:
        """Initialize conv layers of kpt head.
        Kpt head convs are usually different from cls and reg head convs.
        It's deeper.

        """
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.kpt_stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.kpt_stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(
            self
    ) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_kpt = nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1)
        conv_vis = nn.Conv2d(self.feat_channels, self.num_keypoints, 1)
        return conv_kpt, conv_vis

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_vis in self.multi_level_conv_vis:
            conv_vis.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """

        return multi_apply(self.forward_single, x, self.multi_level_kpt_convs,
                           self.multi_level_conv_kpt,
                           self.multi_level_conv_vis)

    def forward_single(
            self, x: Tensor, kpt_convs: nn.Module, conv_kpt: nn.Module,
            conv_vis: nn.Module
    ) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        kpt_feat = kpt_convs(x)
        kpt_pred = conv_kpt(kpt_feat)
        vis_pred = conv_vis(kpt_feat)

        return kpt_pred, vis_pred


@MODELS.register_module()
class YOLOXKptOnlyHead(YOLOv5Head):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        head_module(ConfigType): Base module used for YOLOXHead
        prior_generator: Points generator feature maps in
            2D points-based detectors.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        loss_bbox_aux (:obj:`ConfigDict` or dict): Config of bbox aux loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0,
                     strides=[8, 16, 32]),
                 kpt_coder: ConfigType = dict(type='YOLOXKptCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_kpt: ConfigType = dict(
                     type='mmdet.L1Loss', reduction='sum', loss_weight=1.0),
                 loss_bbox_aux: ConfigType = dict(
                        type='mmdet.L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        self.use_bbox_aux = False
        self.loss_bbox_aux = loss_bbox_aux
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            loss_cls=loss_cls,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_kpt = MODELS.build(loss_kpt)
        self.num_keypoints = head_module['num_keypoints']
        self.kpt_coder = TASK_UTILS.build(kpt_coder)

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        self.loss_bbox_aux: nn.Module = MODELS.build(self.loss_bbox_aux)
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            # YOLOX does not support sampling
            self.sampler = PseudoSampler()

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        return self.head_module(x)

    def predict_by_feat(self,
                        kpt_preds: Optional[List[Tensor]] = None,
                        vis_preds: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """
        predict_by_feat predict feature from feature maps.

        head module outputs are feature maps, so we need to decode them to get the final prediction.

        Args:
            cls_scores (List[Tensor]): feature maps of classification.
            bbox_preds (List[Tensor]): feature maps of localization.
            objectnesses (Optional[List[Tensor]], optional): feature maps of objectness. Defaults to None.
            kpt_preds (Optional[List[Tensor]], optional): feature maps of keypoints. Defaults to None.
            vis_preds (Optional[List[Tensor]], optional): feature maps of keypoints visibility. Defaults to None.
            batch_img_metas (Optional[List[dict]], optional): meta info for images. Defaults to None.
            cfg (Optional[ConfigDict], optional): from config file "model.test_cfg". Defaults to None.
            rescale (bool, optional): whether rescale back to original images. Defaults to True.
            with_nms (bool, optional): whether use nms. Defaults to True.

        Returns:
            List[InstanceData]: prediction results, including boxes, labels, scores, keypoints, keypoints visibility.
        """
        assert len(kpt_preds) == len(vis_preds)
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [kpt_pred.shape[2:] for kpt_pred in kpt_preds]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=kpt_preds[0].dtype,
                device=kpt_preds[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # keypoints and visibility
        flatten_kpt_preds = [
            kpt_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_keypoints * 2)
            for kpt_pred in kpt_preds
        ]
        flatten_kpt_preds = torch.cat(flatten_kpt_preds, dim=1)
        flatten_decoded_kpts = self.kpt_coder.decode(flatten_priors,
                                                     flatten_kpt_preds,
                                                     flatten_stride)
        # flatten bounding boxes based on keypoints.
        flatten_decoded_bboxes = self._kpts2_bbox(flatten_decoded_kpts)

        flatten_vis_preds = [
            vis_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_keypoints)
            for vis_pred in vis_preds
        ]
        flatten_vis_preds = torch.cat(flatten_vis_preds, dim=1).sigmoid()

        flatten_objectness = [None for _ in range(num_imgs)]
        flatten_cls_scores = flatten_vis_preds.max(-1, keepdim=True)[0]

        results_list = []
        for (bboxes, scores, objectness, img_meta, kpts,
             kpts_vis) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas,
                              flatten_decoded_kpts, flatten_vis_preds):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                empty_results.keypoints = kpts
                empty_results.keypoint_scores = kpts_vis
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                keypoints=kpts[keep_idxs],
                keypoint_scores=kpts_vis[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                    results.keypoints -= results.keypoints.new_tensor(
                        [pad_param[2], pad_param[0]])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))
                results.keypoints /= results.keypoints.new_tensor(
                    scale_factor).repeat((1, self.num_keypoints, 1))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            # results = self._kpt_post_process(
            #     results,
            #     cfg,
            #     rescale=False,
            #     with_nms=with_nms,
            #     img_meta=img_meta)
            # keypoints outside the image, the visibility is set to 0
            # results.keypoints[:, :, 0].clamp_(0, ori_shape[1])
            # results.keypoints[:, :, 1].clamp_(0, ori_shape[0])
            # results.keypoint_scores[results.keypoints[:, :, 0] == 0] = 0
            # results.keypoint_scores[results.keypoints[:, :, 1] == 0] = 0

            results_list.append(results)
        return results_list

    def _kpt_post_process(self,
                          results,
                          cfg,
                          rescale=False,
                          with_nms=True,
                          img_meta=None):
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.keypoints = Keypoints._kpt_rescale(results.keypoints,
                                                       scale_factor)
        return results

    def _kpts2_bbox(self, kpts:Tensor, with_vis=True)->Tensor:
        """Convert keypoints to bounding boxes.

        Args:
            kpts (Tensor): Keypoints with shape (n, k, 2) or (n, k, 3).
            with_vis (bool): Whether keypoints contains visibility flag.

        Returns:
            Tensor: Bounding boxes with shape (n, 4).
        """
        if with_vis:
            kpts = kpts[..., :2]
        x1 = kpts[..., 0].min(dim=-1)[0]
        y1 = kpts[..., 1].min(dim=-1)[0]
        x2 = kpts[..., 0].max(dim=-1)[0]
        y2 = kpts[..., 1].max(dim=-1)[0]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def loss_by_feat(
            self,
            kpt_preds: Sequence[Tensor],
            vis_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [kpt_pred.shape[2:] for kpt_pred in kpt_preds]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=kpt_preds[0].dtype,
            device=kpt_preds[0].device,
            with_stride=True)

        flatten_kpt_preds = [
            kpt_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_keypoints * 2)
            for kpt_pred in kpt_preds
        ]
        flatten_vis_preds = [
            vis_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_keypoints)
            for vis_pred in vis_preds
        ]

        flatten_priors = torch.cat(mlvl_priors)
        flatten_kpt_preds = torch.cat(flatten_kpt_preds, dim=1)
        flatten_kpts = self.kpt_coder.decode(flatten_priors[..., :2],
                                             flatten_kpt_preds,
                                             flatten_priors[..., 2])
        flatten_vis_preds = torch.cat(flatten_vis_preds, dim=1)

        # use keypoints visibility as objectness preds
        flatten_objectness = flatten_vis_preds.max(dim=-1)[0]

        (pos_masks, cls_targets, obj_targets, bbox_targets, kpt_targets,
         vis_targets, bbox_aux_target, num_fg_imgs) = multi_apply(
             self._get_targets_single,
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1), flatten_kpts.detach(),
             flatten_objectness.detach(), batch_gt_instances, batch_img_metas,
             batch_gt_instances_ignore)

        # The experimental results show that 'reduce_mean' can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_kpt_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        kpt_targets = torch.cat(kpt_targets, 0)
        vis_targets = torch.cat(vis_targets, 0)

        kpt_mask = vis_targets > 0
        if self.use_bbox_aux:
            bbox_aux_target = torch.cat(bbox_aux_target, 0)

        # loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
        #                          obj_targets) / num_total_samples
        if num_pos > 0:
            # combine kpt and vis for oks loss to N x K x 3.
            flatten_kpt_vis_preds = torch.cat(
                [flatten_kpts, flatten_vis_preds[..., None]], dim=-1)
            kpt_vis_targets = torch.cat([kpt_targets, vis_targets[..., None]],
                                        dim=-1)

            # filter by keypoint visibility.
            # flatten_kpt_vis_preds = flatten_kpt_vis_preds[pos_masks][kpt_mask]
            # kpt_vis_targets = kpt_vis_targets[pos_masks][kpt_mask]
            loss_kpt = self.loss_kpt(
                flatten_kpts.view(-1, self.num_keypoints, 2)[pos_masks],
                kpt_targets, kpt_mask, bbox_targets)
            # loss_kpt = self.loss_kpt(
            #     flatten_kpt_vis_preds.view(-1, self.num_keypoints, 3)[pos_masks], kpt_vis_targets, kpt_mask)
            loss_vis = self.loss_cls(
                flatten_vis_preds.view(-1, self.num_keypoints)[pos_masks],
                kpt_mask.float()) / kpt_mask.sum()
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_kpt = flatten_kpt_preds.sum() * 0
            loss_vis = flatten_vis_preds.sum() * 0

        loss_dict = dict(
            loss_kpt=loss_kpt,
            loss_vis=loss_vis)

        if self.use_bbox_aux:
            if num_pos > 0:
                loss_bbox_aux = self.loss_bbox_aux(
                    flatten_kpt_preds.view(-1, self.num_keypoints, 2)[pos_masks],
                    kpt_targets) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_bbox_aux = flatten_kpt_preds.sum() * 0
            loss_dict.update(loss_bbox_aux=loss_bbox_aux)

        return loss_dict

    @torch.no_grad()
    def _get_kpt_targets_single(
            self,
            kpt_preds: Tensor,
            vis_preds: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            gt_instances_ignore: Optional[InstanceData] = None) -> tuple:
        """Return gt keypoints and visibilities for each prior."""
        # get the ground truth of keypoints
        gt_kpts = gt_instances['keypoints']
        if gt_kpts.shape[0] == 0:
            kpt_targets = gt_kpts.new_zeros((0, self.num_keypoints * 2))
            vis_targets = gt_kpts.new_zeros((0, self.num_keypoints))
            return kpt_targets, vis_targets
        else:
            kpt_targets = gt_kpts[..., :2]
            vis_targets = gt_kpts[..., 2]
            return kpt_targets, vis_targets

    @torch.no_grad()
    def _get_targets_single(
            self,
            priors: Tensor,
            decoded_kpts: Tensor,
            objectness: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            gt_instances_ignore: Optional[InstanceData] = None) -> tuple:
        """Compute classification, regression, and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            tuple:
                foreground_mask (list[Tensor]): Binary mask of foreground
                targets.
                cls_target (list[Tensor]): Classification targets of an image.
                obj_target (list[Tensor]): Objectness targets of an image.
                bbox_target (list[Tensor]): BBox targets of an image.
                bbox_aux_target (int): BBox aux targets of an image.
                num_pos_per_img (int): Number of positive samples in an image.
        """

        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = objectness.new_zeros((0, self.num_classes))
            bbox_target = objectness.new_zeros((0, 4))
            kpt_target = objectness.new_zeros((0, self.num_keypoints, 2))
            vis_target = objectness.new_zeros((0, self.num_keypoints))
            bbox_aux_target = objectness.new_zeros((0, 4))
            obj_target = objectness.new_zeros((num_priors, 1))
            foreground_mask = objectness.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    kpt_target, vis_target, bbox_aux_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = objectness.unsqueeze(1).sigmoid()
        decoded_bbox = self._kpts2_bbox(decoded_kpts, with_vis=False)


        pred_instances = InstanceData(
            bboxes=decoded_bbox, scores=scores, priors=offset_priors)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        bbox_aux_target = objectness.new_zeros((num_pos_per_img, 4))
        if self.use_bbox_aux:
            bbox_aux_target = self._get_bbox_aux_target(
                bbox_aux_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        # TODO: kps target
        kpt_target = gt_instances['keypoints'][
            sampling_result.pos_assigned_gt_inds]
        vis_target = gt_instances['keypoints_visible'][
            sampling_result.pos_assigned_gt_inds]
        return (foreground_mask, cls_target, obj_target, bbox_target,
                kpt_target, vis_target, bbox_aux_target, num_pos_per_img)

    def _get_bbox_aux_target(self,
                             bbox_aux_target: Tensor,
                             gt_bboxes: Tensor,
                             priors: Tensor,
                             eps: float = 1e-8) -> Tensor:
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        bbox_aux_target[:, :2] = (gt_cxcywh[:, :2] -
                                  priors[:, :2]) / priors[:, 2:]
        bbox_aux_target[:,
                        2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return bbox_aux_target
