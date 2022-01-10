import torch
import torch.nn as nn
import base_models
import torch.nn.functional as F
import numpy as np
import math
from utils import batch_ids2words
from torch.nn.utils.rnn import pack_padded_sequence


class Img2WordsCNN(nn.Module):
    def __init__(self, 
                 arch,
                 mode, 
                 vocab_size,
                 max_seq_length):
        """Load the pretrained ResNet and replace top fc layer."""
        super(Img2WordsCNN, self).__init__()
        self.mode = mode
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.cnn = base_models.__dict__[arch](pretrained=True, mode=self.mode, vocab_size=self.vocab_size)

    def words_split(self, mul_class):
        batch_size = mul_class.size(0)
        sorts, indices = torch.sort(mul_class, dim=1, descending=True)
        bin_sorts = (sorts*2).int().float()

        words = (torch.ones(batch_size, self.max_seq_length)*2).long().cuda()
        for i in range(batch_size):
            for j in range(self.max_seq_length):
                if bin_sorts[i, j] != 0:
                    words[i, j] = indices[i, j]
                else:
                    break

        return words

    def forward(self, images):
        """Extract feature vectors from input images."""
        mul_class = self.cnn(images)
        words = self.words_split(mul_class)

        return mul_class, words

    def cam(self, images):
        """Extract feature vectors from input images."""
        features = self.cnn.get_features(images)
        batch_size, channel, height, width = features.size()
        features = features.view(batch_size, channel, -1).permute(0,2,1)
        heat_map = self.cnn.fc_cls(features)
        heat_max, _ = torch.max(heat_map, 1)
        heat_min, _ = torch.min(heat_map, 1)
        heat_max = heat_max.view(batch_size, 1, self.vocab_size)
        heat_min = heat_min.view(batch_size, 1, self.vocab_size)
        heat_map = (heat_map - heat_min) / (heat_max - heat_min + 0.0000001)
        heat_map = heat_map.permute(0,2,1).view(batch_size, self.vocab_size, height, width)
        heat_map = F.interpolate(heat_map, size=(224, 224), mode='bilinear')
        return heat_map


class Words2SenTrm(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 vocab_size,
                 num_layers,
                 dim_feedforward, 
                 vocab,
                 max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(Words2SenTrm, self).__init__()
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(d_model=embed_dim,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward)
        self.de_embedding = nn.Linear(embed_dim, vocab_size)

    def forward(self, words, captions):
        """Decode image feature vectors and generates captions."""
        words_embed = self.embedding(words)
        captions_embed = self.embedding(captions)
        words_embed = words_embed.permute(1,0,2)
        captions_embed = captions_embed.permute(1,0,2)        

        tgt_mask = self.transformer.generate_square_subsequent_mask(self.max_seq_length-1).t().cuda()
        sentences = self.transformer(src=words_embed, 
                                     tgt=captions_embed,
                                     tgt_mask=tgt_mask)

        sentences = sentences.permute(1,0,2)
        sentences = self.de_embedding(sentences)

        return sentences
    
    def sample(self, words):
        """Generate captions for given image features using greedy search."""
        words_embed = self.embedding(words)
        words_embed = words_embed.permute(1,0,2)
        memory = self.transformer.encoder(words_embed)

        seq = torch.ones(1,1).type_as(words)
        for i in range(self.max_seq_length-1):
            out_mask = self.transformer.generate_square_subsequent_mask(seq.size(0)).t().cuda()
            out = self.transformer.decoder(self.embedding(seq), memory, out_mask)
            # print ('out: ', out.size())
            prob = self.de_embedding(out[-1, ::])
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.unsqueeze(dim=0)
            # print ('next_word: ', next_word.size(), next_word)
            # print ('seq: ', seq.size())
            # print ()
            seq = torch.cat([seq, next_word], dim=0)

        seq = seq.permute(1, 0)

        return seq


class Words2SenRNN(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 vocab_size,
                 num_layers,
                 dim_feedforward, 
                 vocab,
                 max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(Words2SenRNN, self).__init__()
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.debedding = nn.Linear(embed_dim, vocab_size)

        self.lstm_encoding = nn.LSTMCell(embed_dim, embed_dim)
        self.lstm_decoding = nn.LSTMCell(embed_dim, embed_dim)
        self.init_h = nn.Linear(embed_dim, embed_dim)
        self.init_c = nn.Linear(embed_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, words, captions):
        """Decode image feature vectors and generates captions."""
        words_embed = self.embedding(words)
        captions_embed = self.embedding(captions)
        sentences = torch.zeros(words.size(0), self.max_seq_length-1, self.vocab_size).cuda()

        for i in range(self.max_seq_length):
            h_embed, c_embed = self.lstm_encoding(words_embed[:, i, :])
        feats_embed = h_embed[:]

        h = self.init_h(feats_embed)
        c = self.init_c(feats_embed)

        for t in range(self.max_seq_length-1):
            inputs = words_embed[:, t, :]
            h,c = self.lstm_decoding(inputs, (h, c))
            pred = self.softmax(self.linear(h))
            sentences[:, t, :] = pred

        return sentences
 
    def sample(self, words):
        """Generate captions for given image features using greedy search."""
        words_embed = self.embedding(words)

        for i in range(self.max_seq_length):
            h_embed, c_embed = self.lstm_encoding(words_embed[:, i, :])
        feats_embed = h_embed[:]

        h = self.init_h(feats_embed)
        c = self.init_c(feats_embed)

        seq = torch.ones(1,1).type_as(words)
        next_word = seq
        for i in range(self.max_seq_length-1):
            h,c = self.lstm_decoding(self.embedding(next_word).squeeze(dim=1), 
                                     (h, c))
            pred = self.softmax(self.linear(h))
            _, next_word = torch.max(pred, dim=-1)
            next_word = next_word.unsqueeze(dim=0)
            seq = torch.cat([seq, next_word], dim=1)

        return seq


class build_model(nn.Module):
    def __init__(self, 
                 arch,
                 mode,
                 vocab,
                 vocab_size,
                 transformer_size,
                 max_seq_length):
        """Load the pretrained ResNet and replace top fc layer."""
        super(build_model, self).__init__()
        if transformer_size == 's1':
            embed_dim = 64
            num_layers = 1
            dim_feedforward = 256
        elif transformer_size == 's2':
            embed_dim = 128
            num_layers = 2
            dim_feedforward = 512
        elif transformer_size == 's3':
            embed_dim = 256
            num_layers = 4
            dim_feedforward = 1024
        elif transformer_size == 's4':
            embed_dim = 512
            num_layers = 8
            dim_feedforward = 2048
        else:
            raise IndexError('No Such Transformer Size')

        self.mode = mode
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.wordsNet = Img2WordsCNN(arch,
                                     mode,
                                     vocab_size,
                                     max_seq_length)
        self.sentNet = Words2SenTrm(embed_dim, 
                                    vocab_size,
                                    num_layers,
                                    dim_feedforward, 
                                    vocab,
                                    max_seq_length)
        if self.mode == 'class_only':
            for i in self.sentNet.parameters():
                i.require_grad = False
        elif self.mode == 'caption_only':
            for i in self.wordsNet.parameters():
                i.require_grad = False

    def get_parameters(self):
        params = list(self.parameters())

        return params

    def forward(self, images, captions):
        if self.mode == 'class_only':
            mul_class, words = self.wordsNet(images) 
            sequences = torch.zeros(images.size(0), self.max_seq_length-1, self.vocab_size).cuda()
        else:
            mul_class, words = self.wordsNet(images) 
            sequences = self.sentNet(words, captions)

        return sequences, mul_class

    def sample(self, image):
        mul_class, words = self.wordsNet(image)

        if self.mode == 'class_only':
            hypothese = (torch.ones(1, self.max_seq_length)*2)
            mul_class = (mul_class*2).int().float() 

            cnt = 0
            for i in range(mul_class.size(1)):
                if mul_class[:, i] != 0:
                    if i!=2 and i!=self.vocab('.'):
                        hypothese[:, cnt] = i
                        cnt += 1
                        if cnt == hypothese.size(1):
                            break

            # add point: ucm:154, sydney:103, rsicd:186
            for i in range(hypothese.size(1)):
                if hypothese[0, i] == 2:
                    hypothese[0, i] = self.vocab('.')
                    break
        else:
            hypothese = self.sentNet.sample(words)

            words = (torch.ones(1, self.max_seq_length)*2)
            mul_class = (mul_class*2).int().float() 

            cnt = 0
            for i in range(mul_class.size(1)):
                if mul_class[:, i] != 0:
                    if i!=2 and i!=self.vocab('.'):
                        words[:, cnt] = i
                        cnt += 1
                        if cnt == words.size(1):
                            break

            # add point: ucm:154, sydney:103, rsicd:186
            for i in range(words.size(1)):
                if words[0, i] == 2:
                    words[0, i] = self.vocab('.')
                    break

        return hypothese, mul_class, words

