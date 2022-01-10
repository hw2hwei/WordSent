import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                            default='/checkpoints/dataset_arch_size_mode.pkl',
                            help='path for trained encoder')
    parser.add_argument('--vocab_path', type=str, default='./data/dataset/vocab.pkl',
                            help='path for vocabulary wrapper')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',  
                            help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',  
                            help='directory for resized images')
    parser.add_argument('--train_caption_path', type=str,
                            default='./data/dataset/annotations/train_dataset.json',
                            help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str,
                            default='./data/dataset/annotations/val_dataset.json',
                            help='path for train annotation json file')
    parser.add_argument('--is_visualize', type=bool, default=False,
                            help='if to visualize the attention map')


    parser.add_argument('--dataset', type=str, default='sydney',
                            choices=['sydney', 'ucm', 'rsicd'])    

    parser.add_argument('--arch', type=str, default='resnet18',
                            choices=['bninception', 'inceptionresnetv2', 'inceptionv3', 
                                    'inceptionv4', 'alexnet', 'resnet18', 'resnet34', 
                                    'resnet50', 'resnet101', 'resnet152', 'vgg16', 
                                    'googlenet'])
    parser.add_argument('--mode', type=str, default="caption_only", 
                            choices=['class_only', 'caption_only'])
    parser.add_argument('--loss_type', type=str, default="MSE_loss", 
                            choices=['L1_loss', 'MSE_loss', 'Hinge_loss', 'BCE_loss'])


    parser.add_argument('--transformer_size', type=str, default='s2',
                            choices=['s1', 's2', 's3', 's4'])
    parser.add_argument('--max_seq_length', type=int , default=30,
                            help='max length of sequence')

    # learning setting
    parser.add_argument('--start_epoch', type=int, default=0,
                            help='start epoch for training')
    parser.add_argument('--end_epoch', type=int, default=50,
                            help='end epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--step_size', type=float, default=50)
    parser.add_argument('--image_size', type=int, default=224)
 
    args = parser.parse_args()
    return args
 
