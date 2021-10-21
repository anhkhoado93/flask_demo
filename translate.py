"""
!pip install pyvi
!pip install https://github.com/trungtv/vi_spacy/raw/master/packages/vi_spacy_model-0.2.1/dist/vi_spacy_model-0.2.1.tar.gz
!gdown https://drive.google.com/uc?id=1kZnu9yI385y78u1IznWwNj5Js6Vep30n
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy.data import Field, BucketIterator
import spacy
from torchtext.legacy import data
from model import Encoder, Decoder, Seq2Seq


spacy_vn = spacy.load('vi_spacy_model')


def load_vocab(path):
    stoi_vocab = dict()
    itos_vocab = dict()
    index = -1
    with open(path, 'r') as f:
        for line in f:
            nline = line[:-1]
            index, token =nline.split('\t',1)
            stoi_vocab[token] = int(index)
            itos_vocab[str(index)] = token
    return [stoi_vocab, itos_vocab]

def load_model(path, src_vocab, trg_vocab): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = len(src_vocab[0])
    OUTPUT_DIM = len(trg_vocab[0])
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)
    SRC_PAD_IDX = src_vocab[0]['<pad>']
    TRG_PAD_IDX = trg_vocab[0]['<pad>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    return model




SRC_VOCAB = load_vocab('resources/vocabs/src.txt')
TRG_VOCAB = load_vocab('resources/vocabs/trg.txt')
MODEL     = load_model('resources/model/tut6-model.pt', src_vocab=SRC_VOCAB, trg_vocab=TRG_VOCAB)
def tokenize_vn(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_vn.tokenizer(text)]

def translate(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('vi_spacy_model')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [src_field[0][token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field[0]['<sos>']]

    for i in range(50):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()

        if pred_token == trg_field[0]['<eos>']:
            break
        trg_indexes.append(pred_token)

    trg_tokens = [trg_field[1][str(i)] for i in trg_indexes]
    return ' '.join(trg_tokens[1:])


def correct(sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lst = tokenize_vn(sentence)
    return translate(sentence, SRC_VOCAB, TRG_VOCAB, MODEL, device)