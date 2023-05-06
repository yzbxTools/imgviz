import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm


class DetErrorVis(object):

    def __init__(self, gt, result):
        if isinstance(gt, str):
            gt = COCO(gt)

        if isinstance(result, str):
            result = COCO(result)

        self.gt = gt
        self.rt = result

        self.stats = {}

    def evaluate_img(self, img_id):
        gt_ann_ids = self.gt.getAnnIds(imgIds=[img_id])
        rt_ann_ids = self.rt.getAnnIds(imgIds=[img_id])

        gt_anns = [self.gt.anns[ann_id] for ann_id in gt_ann_ids]
        rt_anns = [self.rt.anns[ann_id] for ann_id in rt_ann_ids]

        gt_bbox = [ann['bbox'] for ann in gt_anns]
        rt_bbox = [ann['bbox'] for ann in rt_anns]

        iscrowd = [int(ann['iscrowd']) for ann in gt_anns]
        ious = maskUtils.iou(rt_bbox, gt_bbox, iscrowd)

        match_ann_ids = []
        miss_ann_ids = []

        if isinstance(ious, np.ndarray):
            match = np.any(ious > 0.5, axis=0)
            tp = np.count_nonzero(match)
            for idx, flag in enumerate(match):
                if flag:
                    match_ann_ids.append(gt_anns[idx]['id'])
                else:
                    miss_ann_ids.append(gt_anns[idx]['id'])
        else:
            tp = 0
            miss_ann_ids = [ann['id'] for ann in gt_anns]

        total_gt = len(gt_bbox)

        self.stats[img_id] = dict(tp=tp,
                                  totol_gt=total_gt,
                                  totol_rt=len(rt_bbox),
                                  recall=round(tp / total_gt, 4) if total_gt > 0 else 0,
                                  miss=total_gt - tp,
                                  match_ann_ids=match_ann_ids,
                                  miss_ann_ids=miss_ann_ids)

    def evaluate(self):
        img_ids = self.gt.getImgIds()
        for img_id in tqdm(img_ids, desc='statistic each image'):
            if img_id not in self.stats:
                self.evaluate_img(img_id)

    def summarize(self):
        self.evaluate()

        img_ids = self.gt.getImgIds()
        miss_ann_ids = []
        for img_id in tqdm(img_ids, desc='summarize'):
            miss = self.stats[img_id]['miss_ann_ids']
            miss_ann_ids.extend(miss)

        return miss_ann_ids
