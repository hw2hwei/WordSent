import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import pickle
from torch import nn
from model import build_model
from utils import *
from data_loader import build_datasets
from build_vocab import Vocabulary
from validate import validate
from train import train
import pdb
import args_parser
args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

def main():
    best_score = 0

    with open(args.vocab_path.replace('dataset', args.dataset), 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    model = build_model(arch=args.arch,
                        mode=args.mode,
                        vocab=vocab,
                        vocab_size=len(vocab),
                        transformer_size=args.transformer_size,
                        max_seq_length=args.max_seq_length).cuda()
    optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)
    sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('mode', str(args.mode))  \
                                .replace('size', args.transformer_size)    \
                                .replace('arch', args.arch) 
    print (model_path)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))

    # Loss and Optimizer
    caption_criterion = nn.CrossEntropyLoss().cuda()
    classes_criterion = nn.BCELoss().cuda()

    # Custom dataloader
    train_loader = build_datasets(args, vocab)

    print ('Validation Bofore Training: ')
    best_score = validate(args=args,
                         vocab=vocab, 
                         model=model,
                         is_visualize=args.is_visualize)
    print ('bleu: {:.2f}'.format(best_score*100))
    print ('')
    # if args.is_visualize:
    #     pdb.set_trace()

    # Epochs
    print ('Start Training: ')
    for epoch in range(args.start_epoch, args.end_epoch):
        # One epoch's traininginceptionv3
        print ('Train_Epoch_{}: '.format(epoch))
        train(args=args,
              vocab=vocab,
              train_loader=train_loader,
              model=model,
              caption_criterion=caption_criterion, 
              classes_criterion=classes_criterion,
              optimizer=optimizer,
              mode=args.mode,
              loss_type=args.loss_type,
              epoch=epoch)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_score = validate(args=args,
                               vocab=vocab, 
                               model=model,
                               is_visualize=args.is_visualize)
        print ('score: {:.2f}'.format(recent_score*100))

        # # save model
        is_best = recent_score > best_score
        best_score = max(recent_score, best_score)
        # is_best = True
        if is_best:
            torch.save(model.state_dict(), model_path)
            print ('Saved!')
            print ('')
        sche.step()

if __name__ == '__main__':
    main()
