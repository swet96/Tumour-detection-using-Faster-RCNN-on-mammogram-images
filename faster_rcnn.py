import argparse
from pprint import pprint

#custom
from utils import set_seed, device, load_checkpoint_model
from model import CUSTOM_FASTER_RCNN
from engine_ddsm import ENGINE_DDSM


parser = argparse.ArgumentParser()

parser.add_argument('--train_img_dir', type=str, default='/home/sweta/scratch/datasets/DDSM_DATA2/Train/images', help="directory where the training images are present")
parser.add_argument('--train_annotate_dir', type=str, default='/home/sweta/scratch/datasets/DDSM_DATA2/Train/Annotations', help="directory where the training annotations are present")

parser.add_argument('--val_img_dir', type=str, default='/home/sweta/scratch/datasets/DDSM_DATA2/Validation/images', help="directory where the validation images are present")
parser.add_argument('--val_annotate_dir', type=str, default='/home/sweta/scratch/datasets/DDSM_DATA2/Validation/Annotations', help="directory where the validation data annotations are present")

parser.add_argument('--test_img_dir', type=str, default='/home/sweta/scratch/datasets/DDSM_DATA2/Test/images', help="directory where the Test images are present")
parser.add_argument('--test_annotate_dir', type=str, default='/home/sweta/scratch/datasets/DDSM_DATA2/Test/Annotations', help="directory where the test data annotations are present")

parser.add_argument('--ckpt_dir', type=str, default='/home/sweta/scratch/FASTER_RCNN/CHECKPOINTS', help="directory where to save the checkpoints")
parser.add_argument('--plot_dir', type=str, default='/home/sweta/scratch/FASTER_RCNN/PLOTS', help="directory where to save all the plots")

parser.add_argument('--visualise', action='store_true', default=True, help='to plot all the graphs')
parser.add_argument('--training', action='store_true', default=True, help="to make inference on a held-out dataset using the trained model")

#loss functions to use
parser.add_argument('--closs', type=str, default="ce", choices=["ce", "focal"], help="which classification loss to use at the fast rcnn predictor")
parser.add_argument('--regloss', type=str, default="mse", choices=["mse", "huber"], help="which regression loss to use at the fast rcnn predictor")

#info about the data
parser.add_argument('--num_classes', default=2, type=int, help="number of classes present in the dataset") #10 for CIFAR10
parser.add_argument('--mine_negatives', action='store_true', default=True)

parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet18'])
parser.add_argument('--backbone_fine_tune_layers', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', action='store_true', default=True)

parser.add_argument('--sched_name', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingLR','MultiStepLR'], help = "which scheduler to use")

parser.add_argument('--train_batch_size', default=8, type=int)
parser.add_argument('--test_batch_size', default=1, type=int)

parser.add_argument('--opt_name', type=str, default='Adam', choices=["Adam", 'RMSProp'], help = "which optimiser to use")
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (default: 0.0005)')#0.05
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate  (default: 0.01)') #0.1

args = parser.parse_args()
pprint(vars(args))


set_seed(args.seed)
DEVICE = device(args.cuda)

custom_fatser_rcnn = CUSTOM_FASTER_RCNN(args)
model = custom_fatser_rcnn.create_model()
model =model.to(DEVICE)

if args.training: 
    engine = ENGINE_DDSM(args, model, DEVICE)
    engine.train(val_full_data =False)
    # engine.get_mAP_pycocoutils(engine.model, 'Validation' )

    # precisions, recalls, mAP   = engine.find_mAP( engine.model, 'Train', full_data= False )
    engine.find_mAP( engine.model, 'Train' )
    engine.find_mAP( engine.model, 'Validation')

    if args.visualise:
        engine.plot()
        
if not args.training:
    model = load_checkpoint_model(args.ckpt_dir, f'checkpoint_backbone_trainable{args.backbone_fine_tune_layers}', model)
    engine.find_mAP(model, 'Test')
    if args.visualise:
        engine.plot()
    


































# backbone = resnet18(pretraine= True)
# def backbone_fine_tune(model, fine_tune_layers):

#     """
#     Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

#     :param fine_tune_layers: How many convolutional layers to be fine-tuned (negative value means all)
#     """
#     for p in model.parameters():
#         p.requires_grad = False
#     # specific to resnet, since second last layer is avgpool layer is there which does not have trainable weights
#     if fine_tune_layers >1:
#         fine_tune_layers +=1

#     # Last convolution layers to be fine-tuned
#     for c in list(model.children())[
#                 0 if fine_tune_layers < 0 else len(list(model.children())) - (fine_tune_layers):]:
#         for p in c.parameters():
#             p.requires_grad = True
#     return model






