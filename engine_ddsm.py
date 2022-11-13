import numpy as np
from time import time

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn as nn
import torch

#custom
from dataset import DDSM_dataset  
from utils import collate_fn, progress_bar, save_checkpoint_model 
from mAP_utils import mean_average_precision, bb_iou
from plot import val_acc_lr_curve, loss_epochs_curve, PR_curve

''' 
targets=({'boxes': tensor([[280.6154, 418.2588, 363.2137, 465.3177]]), 'labels': tensor([1]), 'image_id': tensor(0), 'iscrowd': tensor([0]), 'area': tensor([3886.9783])},)
if batch size is greater than 1, then more dictionaries are appended at the end of the tuple

targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
targets =[{'boxes': tensor([[280.6154, 418.2588, 363.2137, 465.3177]]), 'labels': tensor([1]), 'image_id': tensor(0), 'iscrowd': tensor([0]), 'area': tensor([3886.9783])}]


outputs= [{'boxes': tensor([[ 41.2387,   2.1246, 488.1640, 102.2746],
        [ 15.5864,  16.6927, 152.9982, 268.3601],
        [ 43.8885,  14.5793, 383.6537, 172.5721],
        [ 41.6457,  43.4667, 110.7605, 176.8473]], device='cuda:0'), 'labels': tensor([1, 1, 1,1], device='cuda:0'), 
        'scores': tensor([0.7650, 0.7365, 0.7300, 0.6863], device='cuda:0')}]
if batch size is greater than 1, then more dictionaries are appended at the end of the list

'''



class ENGINE_DDSM(nn.Module):
    def __init__(self, args, model, DEVICE):
        super(ENGINE_DDSM, self).__init__()
        self.args = args
        self.model = model
        self.optimizer= self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.device = DEVICE
        self.training_loss_list= []
        self.val_acc_list =[]
        self.lr_list= []
        self._print_params()
        self.best_val_mAP = 0

    def _print_params(self):
        print("\n")
        total_params = sum( p.numel() for p in self.model.parameters() )
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum( p.numel() for p in self.model.parameters() if p.requires_grad )
        print(f'{total_trainable_params:,} training parameters.')
        
    def _get_optimizer(self):
        optimizer_args ={'lr':self.args.lr, 'weight_decay':self.args.weight_decay}

        if self.args.opt_name.lower() == 'sgd':
            optimizer_args["momentum"] =  self.args.momentum
        if self.args.opt_name.lower() == 'adam':
            pass
        if self.args.opt_name.lower() == 'rmsprop':
            pass
        else:
            NotImplementedError 
        return getattr(optim, self.args.opt_name)(self.model.parameters(), **optimizer_args)

    def _get_scheduler(self):
        scheduler_args = {}
        if self.args.sched_name == "CosineAnnealingLR":
            scheduler_args["T_max"] =  self.args.num_epochs
        elif self.args.sched_name == "MultiStepLR":
            scheduler_args["milestones"] = [e for e in range(0,self.args.num_epochs,3)]
            scheduler_args["gamma"]= 0.5; scheduler_args["verbose"]= True
        else:
            NotImplementedError
        return getattr(lr_scheduler, self.args.sched_name)(self.optimizer, **scheduler_args)

    def _get_dataloader(self, data_type, full_data):
        if data_type== 'Train':
            train_ddsm_dataset = DDSM_dataset(img_dir= self.args.train_img_dir, annotate_dir= self.args.train_annotate_dir, which_transform = 'Train')
            train_loader = DataLoader(train_ddsm_dataset, batch_size=self.args.train_batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn= collate_fn)
            return train_loader
        if data_type== 'Validation':
            val_ddsm_dataset = DDSM_dataset(img_dir= self.args.val_img_dir, annotate_dir= self.args.val_annotate_dir, which_transform = 'Test', full_data=full_data)
            val_loader = DataLoader(val_ddsm_dataset, batch_size=self.args.test_batch_size, drop_last=False, shuffle=False, num_workers=0, collate_fn= collate_fn)
            return val_loader
        if data_type== 'Test':
            test_ddsm_dataset = DDSM_dataset(img_dir= self.args.test_img_dir, annotate_dir= self.args.test_annotate_dir, which_transform = 'Test', full_data= full_data)
            test_loader = DataLoader(test_ddsm_dataset, batch_size=self.args.test_batch_size, drop_last=False, shuffle=False, num_workers=0, collate_fn= collate_fn)

    def _train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch_idx, (images, targets) in enumerate(loader):
            # print(f"{batch_idx}/ {len(loader)}")

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            total_loss +=losses
            [self.loss_dict[k].append(v.detach().cpu().item()) for k,v in loss_dict.items() ]
            
            losses.backward()
            self.optimizer.step()

            if (batch_idx+1) %4 ==0:
                progress_bar(batch_idx, len(loader), 'Classifier Loss: %.3f | Box Reg Loss: %.3f | Objectness Loss: %.3f | RPN Box Reg Loss: %.3f | Total loss: %.3f'
                        % (loss_dict['loss_classifier'].detach().cpu().item(), loss_dict['loss_box_reg'].detach().cpu().item(),
                        loss_dict['loss_objectness'].detach().cpu().item(), loss_dict['loss_rpn_box_reg'].detach().cpu().item(), total_loss/(batch_idx+1) ))
        self.training_loss_list.append(total_loss.cpu().item()/batch_idx+1)

    def train(self, val_full_data=False):
        #we only use the malignant data for training, so full_data=False
        train_loader = self._get_dataloader(data_type= 'Train', full_data=False)
        val_loader = self._get_dataloader(data_type= 'Validation', full_data=val_full_data)

        for epoch in range(self.args.num_epochs):
            self.curr_epoch= epoch+1
            self.loss_dict = {'loss_classifier':[], 'loss_box_reg':[], 'loss_objectness':[], 'loss_rpn_box_reg': []}
            self._train_one_epoch(train_loader)
            self.lr_list.append(self.scheduler.get_last_lr())
            self.scheduler.step()

            print(f"TRAINING: Epoch {epoch+1}/{self.args.num_epochs} finished!")
            for k,v in self.loss_dict.items():
                print(f"{k}: {np.mean(np.array(v))}")
            print("\n \n")

            # final_acc, _,_, mAP= self.find_mAP(self.model, data_type='Validation', full_data=val_full_data, val_loader)

            # if mAP > self.best_val_mAP:
            #     save_checkpoint_model(self.args.ckpt_dir, f'checkpoint_backbone_trainable{self.args.backbone_fine_tune_layers}', self.model)    
            #     self.best_val_mAP = mAP
            save_checkpoint_model(self.args.ckpt_dir, f'checkpoint_backbone_trainable{self.args.backbone_fine_tune_layers}', self.model)  
            
            val_acc= self.find_acc(self.model, val_loader)
            self.val_acc_list.append(val_acc)
            # to plot as each epoch passes
            # self.plot()
      
    def get_mAP_pycocoutils(self, model, data_type, full_data):
        '''Uses coco evaluator from pycocotools '''
        from coco_utils import evaluate
        start = time()
        model.eval()
        dataloader = self._get_dataloader(data_type= data_type)
        evaluate(model, dataloader, self.device)
        print("Time taken to get mAP for pycocoutils is : ", np.round(time()-start,3))

    def prepare_data_for_metric(self,model, data_type, full_data, dataloader=None):
        '''
        To modify the data format for the mean_average_precision function which gives mAP, precisions and recalls, we use the full data for this
        both boxes have the data type [[int, int, float, list]]
        [[image_id, labels, score, [x1,y1,x2,y2]],[[image_id, labels, score, [x1,y1,x2,y2]],....]
        pred_boxes= [[0, 1, 0.7649857997894287, [ 41.2387,   2.1246, 488.1640, 102.2746]],[63,...], ....] and len(pred_boxes)= batch size
        true_boxes= [[0, 1, 1, [280.6154, 418.2588, 363.2137, 465.3177]], [], ...., [63,...],[],....]    and len(true_boxes) = total no of prediction for the images in the batch 
        
        '''
        print("\n")
        print(f"Preapring {data_type} data to get mAP, accuracy, Precisions and recalls......")
        model.eval()
        if not dataloader:
            dataloader = self._get_dataloader(data_type= data_type, full_data= full_data)
        pred_boxes = []
        true_boxes = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):

                images = list(img.to(self.device) for img in images)
                
                targets = [{k: v.detach().to('cpu') for k, v in t.items()} for t in targets]
                for idx in range(len(targets)):
                    # true_boxes.append([targets[idx]['image_id'].item(), targets[idx]['labels'].item(), targets[idx]['labels'].item(), targets[idx]['boxes'][0][0].item(), targets[idx]['boxes'][0][1].item(), targets[idx]['boxes'][0][2].item(), targets[idx]['boxes'][0][3].item()])
                    true_boxes.append([targets[idx]['image_id'].item(), targets[idx]['labels'].item(), targets[idx]['labels'].item(), targets[idx]['boxes'][0].tolist()])


                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                outputs = model(images)
                outputs= [{k: v.detach().to('cpu') for k, v in t.items()} for t in outputs]

                for id in range(len(outputs)):
                    for idx in range(len(outputs[id]['boxes'])):
                        # pred_boxes.append([targets[id]['image_id'].item(), outputs[id]['labels'][idx].item(), outputs[id]['scores'][idx].item(), outputs[id]['boxes'][idx][0].item(), outputs[id]['boxes'][idx][1].item(), outputs[id]['boxes'][idx][2].item(), outputs[id]['boxes'][idx][3].item()])
                        pred_boxes.append([targets[id]['image_id'].item(), outputs[id]['labels'][idx].item(), outputs[id]['scores'][idx].item(), outputs[id]['boxes'][idx].tolist()])

                progress_bar(batch_idx, len(dataloader),'batch_idx/dataloader')
        return pred_boxes, true_boxes

    def find_mAP(self, model, data_type, full_data=False, dataloader=None):
        start = time()
        pred_boxes, true_boxes = self.prepare_data_for_metric(model, data_type, full_data, dataloader)
        print("\n") 
        print(f"Calculating mAP, accuracy, Precisions and recalls for {data_type} data......")
        final_acc, precisions, recalls, mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=self.args.num_classes-1)
        print(f"mAP for {data_type} data at threshold 0.5: {mAP}") 
        print(f"The final accuracy for {data_type} data at threshold 0.5: {final_acc}") 

        print("Time taken to estimate mAP: ", time()-start)

        self.precisions= precisions
        self.recalls= recalls
        self.final_acc= final_acc
        PR_curve(self.args.plot_dir, self.recalls ,self.precisions, fname = f'{data_type}_PR_curve_backbone_trainable{self.args.backbone_fine_tune_layers}')
        return final_acc, precisions, recalls, mAP     

    def find_acc(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            total_positives=0.0
            true_positives=0.0
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                outputs = model(images)
                outputs= [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
                total_positives +=len(dataloader)

                #special case when each image has a single ground truth box, cross check for more general case
                for id in range(len(targets)):
                    iou_list = []
                    true_box= targets[id]['boxes'][0]
                    for pred_box in outputs[id]['boxes']:
                        iou_list.append(bb_iou(pred_box, true_box))

                    if iou_list:
                        if max(iou_list)>=0.5:
                            true_positives +=1.0    
        return true_positives/total_positives

    def plot(self):
        if self.args.training:
            # PR_curve(self.args.plot_dir, self.recalls ,self.precisions, fname = f'train_PR_curve_backbone_trainable{self.args.backbone_fine_tune_layers}')
            val_acc_lr_curve(self.args.plot_dir, self.lr_list, self.val_acc_list , fname = f'val_acc_vs_lr_backbone_trainable{self.args.backbone_fine_tune_layers}')
            loss_epochs_curve(self.args.plot_dir, list(range(self.curr_epoch)), self.training_loss_list, fname = f'training_loss_vs_epoch_curve_backbone_trainable{self.args.backbone_fine_tune_layers}')
        # if not self.args.training:
        #     PR_curve(self.args.plot_dir, self.recalls ,self.precisions, fname = f'test_PR_curve_backbone_trainable{self.args.backbone_fine_tune_layers}')
