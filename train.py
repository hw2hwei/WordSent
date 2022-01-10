import torch
from torch import nn
from utils import to_var, batch_ids2words
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from PIL import Image
from loss import caption_loss, class_loss


def train(args, 
          vocab, 
          train_loader, 
          model, 
          caption_criterion, 
          classes_criterion, 
          mode,
		  optimizer, 
          loss_type,
          epoch):

    model.train()

    total_num = 0
    loss_caption_total = 0.0
    loss_class_total = 0.0
    total_step = len(train_loader)
    for i, (images, classes, captions) in enumerate(train_loader):
        # Set mini-batch dataset
        images = to_var(images)
        captions = to_var(captions)
        classes = to_var(classes)
        batch_size = images.size(0)

        # Forward, Backward and Optimize
        optimizer.zero_grad()

        outputs_captions, outputs_classes = model(images, captions[::, :-1])

        # classes loss
        loss_class = class_loss(loss_type, outputs_classes, classes, classes_criterion)
        # caption loss
        loss_caption = caption_loss(outputs_captions, captions[::, 1:], caption_criterion)

        if mode == 'none':
            loss = loss_caption
        elif mode == 'class_only':            
            loss = loss_class
        elif mode == 'caption_only':            
            loss = loss_caption
        else:
            raise IndexError('No such mode')

        loss.backward()
        optimizer.step()
        loss_caption_total += loss_caption.item()
        loss_class_total += loss_class.item()

        # calculate the metric scores
        # references = batch_ids2words(captions.view(1, -1), vocab)
        # candidates = batch_ids2words(output_captions.view(1, -1), vocab)
        # classes = batch_ids2words(output_classes.view(1, -1), vocab)

        # Print log info
        total_num += 1
        if (i%20==0 and i!=0) or i==len(train_loader)-1:
            print('Epoch [%d/%d], Step [%d/%d], Caption Loss: %.4f, Class Loss: %5.4f'
                  %(epoch, args.end_epoch, i, total_step, 
                    loss_caption_total/total_num, loss_class_total/total_num)) 
