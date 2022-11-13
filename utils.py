
import torch
import numpy as np
import random

import os
import os.path as osp
import time
import sys

from collections import defaultdict, deque
def device(cuda):
  if cuda:
    return torch.device("cuda:7")
  else:
    return torch.device("cpu")

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
#   torch.backends.cudnn.enabled = False
# To check if the groundtruth bboxes exceed the image size

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return list(zip(*batch))



def save_checkpoint_model(CKPT_DIRECTORY, fname, model):
    ckpt_filename = osp.join(CKPT_DIRECTORY, fname+".pth")
    print(f"==> Saving checkpoint at {ckpt_filename}")
    if not os.path.isdir(CKPT_DIRECTORY):
        os.makedirs(CKPT_DIRECTORY)
    torch.save(model.state_dict(), ckpt_filename)   

def load_checkpoint_model(CKPT_DIRECTORY, fname, model):
    ckpt_fname = osp.join(CKPT_DIRECTORY, fname+".pth")
    print(f"==> Loading checkpoint from {ckpt_fname}")
    assert os.path.isdir(CKPT_DIRECTORY), 'Error: no checkpoint directory found!'
    model.load_state_dict(torch.load(ckpt_fname))
    return model



def check_bbox(bbox: list )->list:
    """Check if bbox boundaries are in range 0, 1, bboxes are normalised according to image height and width"""
    for i in range(4):
      if (bbox[i]<0) :
        bbox[i]=0
      elif (bbox[i]>1) :
        bbox[i]=1
    return bbox




_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 15.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

