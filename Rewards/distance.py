# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os.path
import torch
import torch.nn.parallel
import torch.optim
import os
import torch.utils.data as data
import numpy as np
import os.path
import torch
import torch.nn.parallel
import torch.optim
import alexnet
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import os, sys
import numpy
from matplotlib import pyplot
from numpy import array
from scipy import spatial
from scipy.stats.stats import pearsonr



#from __future__ import division




#
#model_names = sorted(name for name in vgg.__dict__
#    if name.islower() and not name.startswith("__")
#                     and name.startswith("vgg")
#                     and callable(vgg.__dict__[name]))
#
#
#parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
#                    choices=model_names,
#                    help='model architecture: ' + ' | '.join(model_names) +
#                    ' (default: vgg19)')
#parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                    help='number of data loading workers (default: 4)')
#parser.add_argument('--epochs', default=300, type=int, metavar='N',
#                    help='number of total epochs to run')
#parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                    help='manual epoch number (useful on restarts)')
#parser.add_argument('-b', '--batch-size', default=128, type=int,
#                    metavar='N', help='mini-batch size (default: 128)')
#parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
#                    metavar='LR', help='initial learning rate')
#parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                    help='momentum')
#parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
#                    metavar='W', help='weight decay (default: 5e-4)')
#parser.add_argument('--print-freq', '-p', default=20, type=int,
#                    metavar='N', help='print frequency (default: 20)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
#parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                    help='evaluate model on validation set')
#parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')
#parser.add_argument('--half', dest='half', action='store_true',
#                    help='use half-precision(16-bit) ')
#parser.add_argument('--save-dir', dest='save_dir',
#                    help='The directory used to save the trained models',
#                    default='save_temp', type=str)


best_prec1 = 0

#change the urls
#---------------------------------------------------------

#pathV0 ="/more/datas/testing/"
pathV ="/datas/alexnetData/testsmall/"
#pathV2 ="/more/datas/testing/v2resized/"
#pathV3 ="/more/datas/testing/finalstage/"
pathV4 ="/datas/alexnetData/someoneelse/"
pathV5 ="/datas/alexnetData/resized/"


#dirs2 = os.listdir( path2 )
#dirs3 = os.listdir( path3 )
#dirs4 = os.listdir( path4 )

#dirsV0 = os.listdir( pathV0 )
dirsV = os.listdir( pathV )
#dirsV2 = os.listdir( pathV2 )
#dirsV3 = os.listdir( pathV3 )
dirsV4 = os.listdir( pathV4 )
dirsV5 = os.listdir( pathV5 )

#---------------------------------------------------------

def main():

    # load a pretrained alexnet model
    model = alexnet.alexnet(True)

    model.cuda()

    #preprocessing
    model.classifier = nn.Sequential(
    )
    preprocess = transforms.Compose([
    transforms.Pad(1),
    transforms.ToTensor()

    ])

    #number of images
    x = numpy.arange(641)

    # the list where we store the distances of the images
    y = []
    yy = []

    # load and resizing the target image
    im = Image.open('/datas/alexnetData/someoneelse/frame5575.bmp')
    box = (0,20,600,480)
    im_finally = im.crop(box)
    im_finally_Resize = im_finally.resize((512,512))

    #convert the target image to feature vector via alexnet model
    img_tensor = preprocess(im_finally_Resize)
    img_tensor.unsqueeze_(0)
    outTarget = torch.autograd.Variable(img_tensor).cuda()
    print(outTarget)
    sys.exit()
    mTarget = model(outTarget).squeeze_(0)


    # load and resizing the input images
    for item in dirsV:
        if os.path.isfile(pathV+item):
            im = Image.open(pathV+item)
            box = (0,20,600,480)
            a = im.crop(box)
            im_Resize = a.resize((512,512))
            im_Resize.save(pathV5+item, 'bmp')
            print('resized: ',item)

    print ('finished loading and resizing images:')

    j = 1
    sum = 0
    for item in dirsV5:
        if os.path.isfile(pathV5+item):
            #convert the input image to feature vector via alexnet model
            im = Image.open(pathV5+item)
            img_tensor = preprocess(im)
            img_tensor.unsqueeze_(0)
            out = torch.autograd.Variable(img_tensor).cuda()
            m = model(out).squeeze_(0)
            #meature the similarity betwean the target image and the input image
            b = cos_sim(m.data, mTarget.data)
            print(j,' ',item,' ',b)
            j+=1
            y.append(b)
    print('finished calculating')
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0.86,0.92)
    pyplot.plot(x,y)
#   showing or saving the similarity graph
    pyplot.show()
#    pyplot.savefig('results/Alex/VGG512.png')
#    sys.exit()

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


def crop(Path, input, height, width):
    k = 0
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            print(i)
            a = im.crop(box)
            a.save(Path+k+'.png',"PNG")
            print(i)
            k += 1



def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
