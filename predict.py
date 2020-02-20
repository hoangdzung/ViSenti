import pickle
import torch 
from scipy.special import softmax 
from preprocessing import * 
from transformers import *
from underthesea import sentiment

bert_model = torch.load('../bert.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

svm_model = pickle.load(open('../svm.pkl', 'rb'))

MAX_LEN = 512

def bert_predict(sent):
    encoded_sent = tokenizer.encode(sent ,add_special_tokens = True)
    if len(encoded_sent) > MAX_LEN:
        encoded_sent = encoded_sent[:MAX_LEN-1]+ encoded_sent[-2:-1]                                                  
    outputs = bert_model(torch.tensor([encoded_sent]))
    return softmax(outputs[0].detach().cpu().numpy())

sent = input("Nhap cau:")
while(len(sent)>0):
    try:
        a = bert_predict(sent)[0]
        print(np.argmax(a))
    except:
        a = np.zeros((3,))
    try:
        b = svm_model.predict_proba([sent])[0]
        print(np.argmax(b))
    except:
        b = np.zeros((3,))
    try:
        c = sentiment(sent)
        print(c)
    except:
        c = np.zeros((3,))
    else:
        c =  np.zeros((3,))
        if c == 'negative':
            c[2] = 1
        elif c == 'positive':
            c[0] = 1
    final = a + b +c

    print(np.argmax(final))
    
    sent = input("Nhap cau:")
