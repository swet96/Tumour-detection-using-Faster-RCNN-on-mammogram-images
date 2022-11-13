from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.resnet import ResNet50_Weights,ResNet18_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import _default_anchorgen

from torchvision.models import resnet18
import torchvision
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torch.nn as nn
# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=0)
       
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()

class CUSTOM_FASTER_RCNN():
    def __init__(self, args):
        super(CUSTOM_FASTER_RCNN, self).__init__()
        self.args = args

    def create_model(self):
        # trainable_backbone_layers can be from ["layer4", "layer3", "layer2", "layer1", "conv1"]
        resnet50_layers = ["layer4", "layer3", "layer2", "layer1", "conv1"]
        # weights_backbone=ResNet50_Weights.IMAGENET1K_V1 OR ResNet50_Weights.DEFAULT, USE DEFAULT TO GET LATEST UPDATED WEIGHTS 
        # model_kwargs = {'rpn_positive_fraction':0.6,  'box_positive_fraction':0.6 }

        if self.args.model=='resnet50':
            model_kwargs = {'box_nms_thresh': 0.6, 'box_fg_iou_thresh':0.4, 'box_bg_iou_thresh':0.6}

            # model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, kwargs= model_kwargs)

            if self.args.backbone_fine_tune_layers:
                model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=self.args.backbone_fine_tune_layers)
            else:
                model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

            in_features = model.roi_heads.box_head.fc7.out_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.args.num_classes)
            
            if not self.args.backbone_fine_tune_layers:
                for p in model.backbone.body.parameters():
                    p.requires_grad = False
            
            # model = fasterrcnn_resnet50_fpn(num_classes = args.num_classes, weights_backbone= ResNet50_Weights.DEFAULT, trainable_backbone_layers=args.backbone_fine_tune_layers)
            # print(f"{resnet50_layers[:args.backbone_fine_tune_layers]} are traineable!")
            # print(model.roi_heads.box_head)
            # backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18',weights=ResNet18_Weights.DEFAULT, trainable_layers=1 )
            # model = FasterRCNN(backbone,num_classes=2)   
        torchvision.models.detection.roi_heads.fastrcnn_loss= self.custom_fastrcnn_loss
        model.transform.image_mean = [0.3204, 0.3204, 0.3204]
        model.transform.image_std = [0.2557, 0.2557, 0.2557]
        # print(model)
        return model

    def custom_fastrcnn_loss(self,class_logits, box_regression, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.

        Args:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        if self.args.closs=='ce':
            classification_loss = F.cross_entropy(class_logits, labels)
        if self.args.closs=='focal':
            classification_loss = FocalLoss()(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        if self.args.regloss =="mse":
            box_loss = F.mse_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                reduction="sum",
            )
        if self.args.regloss=='huber':
            box_loss = F.huber_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                reduction="sum",
            )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss







        

# from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
# def fasterrcnn_resnet18_fpn(num_classes, weights_backbone, trainable_backbone_layers,progress, **kwargs):
#     norm_layer = misc_nn_ops.FrozenBatchNorm2d
#     backbone = resnet18(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
#     backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
#     model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
#     return model
    

#  elif args.model =='resnet18':
#         resnet18_layers = ["layer4", "layer3", "layer2", "layer1", "conv1"]
#         model_kwargs= {'min_size': 900, 'max_size': 1500}
#         model =fasterrcnn_resnet18_fpn(num_classes=args.num_classes, weights_backbone=None, trainable_backbone_layers=args.backbone_fine_tune_layers, progress=True, **model_kwargs)
#         print(f"{resnet18_layers[:args.backbone_fine_tune_layers]} are traineable!")
#         # backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).features
#         # backbone.out_channels = 512
#         # anchor_generator = _default_anchorgen()# AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
#         # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
#         # model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
#         print(model)