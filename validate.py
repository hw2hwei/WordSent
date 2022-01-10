from torch import nn
from utils import *
from build_vocab import Vocabulary
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocotools.coco import COCO
from PIL import Image
import nltk
import cv2
import pdb
import torchvision.transforms as transforms

def att_visualization(img_path,                               
                      model,
                      img,
                      output,
                      hypotheses):
    output = output.view(-1).detach().cpu().numpy()
    heat_map = model.wordsNet.cam(img).squeeze(dim=0).permute(1,2,0).detach().cpu().numpy()
    hypotheses = hypotheses[0].split(' ')

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    heat_map = cv2.resize(heat_map, (img.shape[0], img.shape[1]))
    save_dir = 'att_visual/' + img_path.split('/')[4].strip('.tif')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create a series of attentioned images
    t = 0
    img_save_path = save_dir + '/' + 'image' + '.jpg'
    cv2.imwrite(img_save_path, img)
    for word in hypotheses:
        if output[t] == 2:
            break
        heat_map_t = heat_map[:, :, output[t]]
        heat_map_t = cv2.applyColorMap(np.uint8(255*heat_map_t), cv2.COLORMAP_JET)
        img_save_path = save_dir + '/' + str(t) + '_' + word + '.jpg'
        img_save = heat_map_t*0.5 + img * 0.5
        cv2.imwrite(img_save_path, img_save)
        t += 1
    # pdb.set_trace()


def validate(args, vocab, model, is_visualize=False):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    bleu_scorer = Bleu(n=4)
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    meteor_scorer = Meteor()

    val_dir = args.val_dir.replace('dataset', args.dataset)
    val_coco = COCO(args.val_caption_path.replace('dataset', args.dataset))
    val_ids = list(val_coco.anns.keys())
    transform = transforms.Compose([ 
        transforms.Resize(args.image_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    captions = {} 
    hypotheses = {}  
    words = {}
    imgs_num = int(len(val_ids)/5)    
    for i in range(imgs_num):
        img_id = val_coco.anns[val_ids[i*5]]['image_id']
        img_path = val_coco.loadImgs(img_id)[0]['file_name']
        print ('image_path: ', img_path)
        image = Image.open(os.path.join(val_dir, img_path)).convert('RGB')
        image = transform(image)
        image = image.cuda()
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        
        output, _, word = model.sample(image)
        word = word.cpu()

        hypothese_i = batch_ids2words(output, vocab)
        word_i = batch_ids2words(word, vocab)

        hypotheses[str(img_id)] = hypothese_i
        words[str(img_id)] = word_i

        if is_visualize:
            att_visualization(os.path.join(val_dir, img_path), 
                              model,
                              image,
                              output,
                              hypotheses)

        print ('hypothese_i: ', hypothese_i)
        captions_i = []
        for ann_id in range(img_id*5, (img_id+1)*5):
            caption = val_coco.anns[ann_id]['caption']   
            captions_i.append(caption)
            print ('caption: ', caption)
        captions[str(img_id)] = captions_i
        print ()

    if args.mode == "class_only":
        (bleu1, bleu2, bleu3, bleu4), _ = bleu_scorer.compute_score(captions, hypotheses)
        print ('belu1: ', bleu1)

        return bleu1

    else:
        (bleu1, bleu2, bleu3, bleu4), _ = bleu_scorer.compute_score(captions, hypotheses)
        cider, _ = cider_scorer.compute_score(captions, hypotheses)
        rouge, _ = rouge_scorer.compute_score(captions, hypotheses)
        meteor, _ = meteor_scorer.compute_score(captions, hypotheses)
        score_avg = (bleu1 + bleu2 + bleu3 + bleu4 + cider/3.0 + rouge + meteor) / 7

        print ('bleu1: ', bleu1)
        print ('bleu2: ', bleu2)
        print ('bleu3: ', bleu3)
        print ('bleu4: ', bleu4)
        print ('cider: ', cider)
        print ('rouge: ', rouge)
        print ('meteor: ', meteor)

        return score_avg