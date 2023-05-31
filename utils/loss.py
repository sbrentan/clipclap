import torch
from torchvision.ops.boxes import box_area
import torch.nn.functional as F

class Loss():
    def xywh2xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)


    def xyxy2xywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)


    def box_iou(boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = Loss.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

    def clipclap_loss(pred, target):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        batch_size = pred.shape[0]
        # world_size = get_world_size()
        num_boxes = batch_size

        # print(pred)
        # print(target)
        loss_bbox = F.l1_loss(pred, target, reduction='none')
        loss_giou = 1 - torch.diag(Loss.generalized_box_iou(
            Loss.xywh2xyxy(pred),
            Loss.xywh2xyxy(target)
        ))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses