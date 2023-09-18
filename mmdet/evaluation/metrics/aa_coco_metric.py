# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable
from mmdet.evaluation.metrics.aa_cocoeval import COCOeval
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls
from .wm_coco_metrics_utils import ap_per_class

@METRICS.register_module()
class wm_CocoMetric(BaseMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        # self.metrics = metric if isinstance(metric, list) else [metric]
        # allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        # allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        # for metric in self.metrics:
        #     if metric not in allowed_metrics:
        #         raise KeyError(
        #             "metric should be one of 'bbox', 'segm', 'proposal', "
        #             f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            gt = dict()
            gt_result = data_sample['gt_instances']
            gt['bboxes'] = gt_result['bboxes'].cpu().numpy()
            gt['labels'] = gt_result['labels'].cpu().numpy()
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        tps_box = []
        scores = []
        labels = []
        gt_boxes = []
        gt_clses = []
        # prs_50,res_50, f1ps_50 = [],[],[]
        for gt, result in results:
            # tp_box, pr_50, re_50, f1p_50 = self.gen_tp(result['bboxes'], gt['bboxes'])
            tp_box = self.gen_tp(result['bboxes'], gt['bboxes'])

            tps_box.append(tp_box)

            scores.append(result['scores'])
            labels.append(result['labels'])
            gt_clses.append(gt['labels'])

        tps_box = np.concatenate(tps_box,0)
        scores = np.concatenate(scores,0)
        labels = np.concatenate(labels,0)
        gt_clses = np.concatenate(gt_clses,0)

        p, r, ap, f1, ap_class = ap_per_class(tp=tps_box, conf=scores, pred_cls=labels, target_cls=gt_clses)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        eval_results = OrderedDict()
        eval_results['precision'] = mp
        eval_results['recall'] = mr
        eval_results['map50'] = map50
        eval_results['map'] = map
        eval_results['F1'] = f1.mean()
        return eval_results

    def gen_tp(self, boxes, gt_boxes):
        cq = 0
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        l = len(boxes)
        tp = np.zeros([l, 10], dtype=np.bool_)
        for gt_box in gt_boxes:
            for idx, box in enumerate(boxes):
                for jdx, threshold in enumerate(thresholds):
                    iou = self.bbox_iou(box, gt_box)
                    if iou>=threshold:
                        tp[idx, jdx] = True
        return tp

    def bbox_iou(self, box1, box2):
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1)
        area2 = box_area(box2)

        inter = 0
        # 基础练习 矩形面积交
        a1 = max(min(box1[0], box1[2]), min(box2[0], box2[2]))
        a2 = min(max(box1[0], box1[2]), max(box2[0], box2[2]))
        b1 = max(min(box1[1], box1[3]), min(box2[1], box2[3]))
        b2 = min(max(box1[1], box1[3]), max(box2[1], box2[3]))
        if a1 < a2 and b1 < b2:
            inter = (a2 - a1) * (b2 - b1)
        return inter / (area1 + area2 - inter)  # iou = inter / (area1 + area2 - inter)


