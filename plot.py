import os.path as osp
import os
import matplotlib.pyplot as plt


def PR_curve(IMAGE_DIRECTORY, x ,y, fname = 'PR_curve'):
    # x= x.detach().cpu().numpy()
    # y= y.detach().cpu().numpy()
    image_fname = osp.join(IMAGE_DIRECTORY,fname+".png")

    print(f"==> Saving PR curve at {image_fname}")
    if not os.path.isdir(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    fig, ax = plt.subplots()
    #linewidth=2, markersize=12, line_style= "dashed"
    ax.plot(x, y,'r>--',ms=4, mfc='b',lw=2 )
    ax.set_title("precision vs recall at 0.5 IoU")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    plt.savefig(image_fname)


def loss_epochs_curve(IMAGE_DIRECTORY, x ,y, fname = 'training_loss_vs_epoch_curve'):
    image_fname = osp.join(IMAGE_DIRECTORY,fname+".png")

    print(f"==> Saving training loss vs epoch curve at {image_fname}")
    if not os.path.isdir(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    fig, ax = plt.subplots()
    #linewidth=2, markersize=12, line_style= "dashed"
    ax.plot(x, y,'g+--',ms=4, mfc='b',lw=2 )
    ax.set_title("loss vs epoch")
    ax.set_xlabel("epoch number")
    ax.set_ylabel("loss")
    # ax.legend(loc= "lower right")
    plt.savefig(image_fname)

def val_acc_lr_curve(IMAGE_DIRECTORY, x ,y, fname = 'val_acc_vs_lr'):
    
    image_fname = osp.join(IMAGE_DIRECTORY,fname+".png")
    print(f"==> Saving val_acc vs lr curve at {image_fname}")
    if not os.path.isdir(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    fig, ax = plt.subplots()
    #linewidth=2, markersize=12, line_style= "dashed"
    ax.plot(x, y,'bo--',ms=4, mfc='b',lw=2 )
    ax.set_title("val_acc vs lr")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("validation accuracy")
    plt.savefig(image_fname)