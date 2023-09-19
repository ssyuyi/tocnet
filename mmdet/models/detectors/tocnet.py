# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple, Union

import torch
from torch import Tensor
import random
from mmdet.structures import OptSampleList, SampleList
from .base import BaseDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

@MODELS.register_module()
class tocnet(BaseDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        res = self.gen_ar(batch_inputs, batch_data_samples)
        x, a, r = res[0], res[1], res[2]
        x = self.backbone_et(x)
        a = self.backbone_et(a)
        r = self.backbone_et(r)
        nfs = self.neck_et(x)
        losses = self.bbox_head.loss(nfs, batch_data_samples)
        loss_ar = self.arloss(x,a,r)
        losses['loss_ar'] = loss_ar*0.1
        return losses

    def gen_ar(self, batch_inputs: Tensor,
             batch_data_samples: SampleList):
        x = copy.deepcopy(batch_inputs)
        a = copy.deepcopy(batch_inputs)
        r = copy.deepcopy(batch_inputs)
        for idx, sample in enumerate(batch_data_samples):
            a, r = self.get_ar(x, a, r, idx, sample)
        return x,a,r

    def get_ar(self, x, a, r, idx, sx, n=7, s=1):
        label = self.gen_label(n, s, sx)
        n, _ = label.shape
        _, c, h, w = x.shape
        cx = h // n
        cy = w // n
        pics = []
        for i in range(n):
            for j in range(n):
                tmp = x[idx, :, cx * i:cx * (i + 1), cy * j:cy * (j + 1)]
                if label[j, i] == 0:
                    pics.append(tmp)
        random.shuffle(pics)
        k = 0
        for i in range(n):
            for j in range(n):
                if label[j, i] == 0:
                    a[idx, :, cx * i:cx * (i + 1), cy * j:cy * (j + 1)] = pics[k]
                    k += 1
        for i in range(n):
            for j in range(n):
                if label[j, i] == 1:
                    r[idx, :, cx * i:cx * (i + 1), cy * j:cy * (j + 1)] = 0
        return a, r

    def gen_label(self, n, s, sample):
        tars = sample.gt_instances.bboxes
        h, w = sample.batch_input_shape
        label = torch.zeros([n, n])
        grid_h = h/n
        grid_w = w/n
        for tar in tars:
            x1, y1, x2, y2 = tar
            zz = torch.clip((x1/grid_w - s), 0, n - 1)
            zz = torch.floor(zz).int()
            zy = torch.clip((x2/grid_w + s), 0, n - 1)
            zy = torch.ceil(zy).int()
            zs = torch.clip((y1/grid_h - s), 0, n - 1)
            zs = torch.floor(zs).int()
            zx = torch.clip((y2/grid_h + s), 0, n - 1)
            zx = torch.ceil(zx).int()
            label[zz:zy, zs:zx] = 1
        return label

    def arloss(self,
               x: Tuple[Tensor],
               a: Tuple[Tensor],
               r: Tuple[Tensor]) -> Tensor:
        losses = []
        for i in range(3):
            loss_tmp = self.sim(x[i], a[i], r[i])
            losses.append(loss_tmp)
        a0 = 0
        a1 = 0
        a2 = 1
        loss_result = losses[0]*a0 + losses[1]*a1 + losses[2]*a2
        return loss_result

    def sim(self, x, a, r):
        x = torch.nn.functional.adaptive_max_pool2d(x,[1,1])
        a = torch.nn.functional.adaptive_max_pool2d(a,[1,1])
        r = torch.nn.functional.adaptive_max_pool2d(r,[1,1])
        res = torch.cosine_similarity(x,r,1) \
              + torch.cosine_similarity(a,r,1) \
              +1-torch.cosine_similarity(x,a,1)
        res = torch.flatten(res)
        res = torch.mean(res)
        return res

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x


    def backbone_et(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        return x

    def neck_et(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.neck(batch_inputs)
        return x


@MODELS.register_module()
class TOCNet(tocnet):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
