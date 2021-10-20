def read_vocab(path):
    vocab = dict()
    index = -1
    with open(path, 'r') as f:
        for line in f:
            nline = line[:-1]
            index, token =nline.split('\t',1)
            vocab[token] = int(index)
    return vocab

SRC_VOCAB = read_vocab('resources/vocabs/src.txt')
TRG_VOCAB = read_vocab('resources/vocabs/trg.txt')

def tokenize(sentence):
    return sentence.split(' ')

def translate(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('vi_spacy_model')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return ' '.join(trg_tokens[1:])

    
def correct(sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lst = tokenize(sentence)
    return translate(sentence, SRC_VOCAB, SRC_VOCAB, model, device)