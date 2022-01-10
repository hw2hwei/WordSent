import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import build_model
from utils import *
from data_loader import build_datasets
from build_vocab import Vocabulary
from collections import OrderedDict 
from validate import validate
from train import train
from calc_scores.bleu import BLEU
from calc_scores.cider import Cider
from calc_scores.rouge import Rouge
# from calc_scores.meteor import Meteor
from pycocotools.coco import COCO
from PIL import Image
import nltk
import pdb

import args_parser
args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    best_bleu = 0

    with open(args.vocab_path.replace('dataset', args.dataset), 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    model = build_model(arch=args.arch,
                        mode=args.mode,
                        vocab=vocab,
                        vocab_size=len(vocab),
                        transformer_size=args.transformer_size,
                        max_seq_length=args.max_seq_length).cuda()

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('mode', str(args.mode))  \
                                .replace('size', args.transformer_size)    \
                                .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=True)
        print ('Load the chekpoint of {}'.format(model_path))


    # Custom dataloader
    print ('Validation Bofore Training: ')
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    bleu1_total = 0.0
    bleu2_total = 0.0
    bleu3_total = 0.0
    bleu4_total = 0.0
    cider_total = 0.0
    rouge_total = 0.0

    val_dir = args.val_dir.replace('dataset', args.dataset)
    val_coco = COCO(args.val_caption_path.replace('dataset', args.dataset))
    val_ids = list(val_coco.anns.keys())
    transform = transforms.Compose([ 
        transforms.Resize(args.image_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    imgs_num = int(len(val_ids)/5)    
    for i in range(imgs_num):
        img_id = val_coco.anns[val_ids[i*5]]['image_id']
        img_path = val_coco.loadImgs(img_id)[0]['file_name']
        # print ('image_path: ', img_path)
        image = Image.open(os.path.join(val_dir, img_path)).convert('RGB')
        image = transform(image)
        image = image.cuda()
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        
        output, multi_classes, words = model.sample(image)
        hypothese = batch_ids2words(output, vocab)

        captions = []
        for ann_id in range(img_id*5, (img_id+1)*5):
            caption = val_coco.anns[ann_id]['caption']   
            captions.append([caption])

        # print (hypothese)
        bleu1 = BLEU(hypothese, captions, 1)
        bleu2 = BLEU(hypothese, captions, 2)
        bleu3 = BLEU(hypothese, captions, 3)
        bleu4 = BLEU(hypothese, captions, 4)
        cider = Cider(hypothese, captions)
        rouge = Rouge(hypothese, captions)
        # meteor = Meteor(hypothese, captions)
        bleu1_total += bleu1
        bleu2_total += bleu2
        bleu3_total += bleu3
        bleu4_total += bleu4
        cider_total += cider
        rouge_total += rouge

    bleu1 = bleu1_total/(imgs_num*1.0)
    bleu2 = bleu2_total/(imgs_num*1.0)
    bleu3 = bleu3_total/(imgs_num*1.0)
    bleu4 = bleu4_total/(imgs_num*1.0)
    cider = cider_total/(imgs_num*1.0)
    rouge = rouge_total/(imgs_num*1.0)
    print ('bleu1: {:.2f}, bleu2: {:.2f}, bleu3: {:.2f}, bleu4: {:.2f}, cider: {:.4f}, rouge: {:.4f}'\
            .format(bleu1*100, bleu2*100, bleu3*100, bleu4*100, cider, rouge))

if __name__ == '__main__':
    main()
