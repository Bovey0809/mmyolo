# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones.res2net import Bottle2neck
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.backbones.resnext import Bottleneck as BottleneckX
from mmdet.models.layers import SimplifiedBasicBlock
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm


def is_block(modules):
    """Check if is ResNet building block."""
    return isinstance(
        modules,
        (
            BasicBlock,
            Bottleneck,
            BottleneckX,
            Bottle2neck,
            SimplifiedBasicBlock,
        ),
    )


def is_norm(modules):
    """Check if is one of the norms."""
    return isinstance(modules, (GroupNorm, _BatchNorm))


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    return not any(
        isinstance(mod, _BatchNorm) and mod.training != train_state
        for mod in modules
    )
