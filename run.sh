python faster_rcnn.py --num_epochs 25 --sched_name 'MultiStepLR' --closs 'ce' --regloss 'mse' --backbone_fine_tune_layers 0 #change it to 1 or 2


python faster_rcnn.py --num_epochs 25 --sched_name 'MultiStepLR' --closs 'ce' --regloss 'huber'

python faster_rcnn.py --num_epochs 25 --sched_name 'MultiStepLR' --closs 'focal' --regloss 'mse'

python faster_rcnn.py --num_epochs 25 --sched_name 'MultiStepLR' --closs 'focal' --regloss 'huber'