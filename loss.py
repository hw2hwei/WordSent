import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def MSE_loss(output, label):
	loss = ((output - label)**2)
	loss = loss.mean()

	return loss

def L1_loss(output, label):
	loss = torch.abs(output - label)
	loss = loss.mean()

	return loss
    

def Hinge_loss(output, label):
	output = output*2 - 1 
	label = label*2 - 1 

	loss = 1 - label*output
	loss = loss.mean()

	return loss

def BCE_loss(output, label):
	loss = -(label*torch.log(output+0.0001) + (1-label)*torch.log(1-output + 0.0001))
	loss = loss.mean()

	return loss


def class_loss(loss_type, output, label, classes_criterion):
    # print (output_classes.size(), output_classes)
    # print (label_classes.size(), label_classes)
    if loss_type == 'BCE_loss':
    	loss = BCE_loss(output, label)
    elif loss_type == 'MSE_loss':
    	loss = MSE_loss(output, label)
    elif loss_type == 'L1_loss':
    	loss = L1_loss(output, label)
    elif loss_type == 'Hinge_loss':
    	loss = Hinge_loss(output, label)
    else:
        raise IndexError('No Such Loss')

    return loss


def caption_loss(output_captions, label_captions, caption_criterion):
    for i in range(label_captions.size(0)):
        loss_caption = caption_criterion(output_captions[i], label_captions[i])  if i==0   \
                    else caption_criterion(output_captions[i], label_captions[i]) + loss_caption
    loss_caption /= label_captions.size(0)

    return loss_caption