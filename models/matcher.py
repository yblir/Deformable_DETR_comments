# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# ������ƥ��
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            # ����ط��Ĵ�����DETR��ͬ,DETR�����ʹ�õ���softmax(1)
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            # DETR�Ĵ�����ʽ
            # ȡ����Ӧ�����ķ���
            # һ��Ԥ����ϵ� ��Ӧ�� �����е�gt�ϵķ���
            # cost_class = -out_prob[:, tgt_ids]


            # ����ط��Ĵ�����DETR��ͬ
            # ������Focal BCELoss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            # p=1 ����l1����
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            # �Ƚ������ĵ���߱�����������ĸ�����ֵ
            # ���еĿ򣬸�gt��giou [bs*300, all_img_gt_count]
            # ��ȫ��ͬλ�õĿ� giou��1����ȫ���ཻ�Ŀ�giou�Ǹ���
            # ����������һ�����ţ���ȫ���ཻ�Ŀ��ֵ�ͱ������������ʾ�˸���Ĵ���
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # ���յĴ��۾���������ƥ��ʹ�õģ�������Ҫ�ܵķ���Ĵ�����С
            # ǰ�涼�Ǹ������Ȩ��ϵ��
            # [bs * proposals num, all_img_gt_count]
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # [bs*100, all_img_gt_count] -> [bs,100,all_img_gt_count]
            # .cpu Ϊ�˸�scipy���� ά�ȱ�Ϊ batch_size , 100, gt������
            C = C.view(bs, num_queries, -1).cpu()
            # ÿ��ͼƬ��Ӧ��gt������
            sizes = [len(v["boxes"]) for v in targets]
            # linear_sum_assignment �����������㷨
            # C.split ����ÿ��ͼƬ��gt�����������з�
            # ��һ��ֵ��proposals��id������100��ȡ��һ���򣬵ڶ���Ӧ���Ƕ�Ӧ����һ��gt��id
            # indices is a list length is bs
            # indices��һ������[(array([ 0, 51]), array([0, 1])), (array([13, 24, 54, 86]), array([0, 1, 3, 2]))]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)