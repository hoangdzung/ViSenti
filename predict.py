import pickle
import torch 
from scipy.special import softmax 
from preprocessing import * 
from transformers import *
from underthesea import sentiment
try:
    bert_model = torch.load('bert.pt')
except:
    pass
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

try:
    svm_model = pickle.load(open('svm.pkl', 'rb'))
except:
    pass

try:
    xgboost_model = pickle.load(open('model_xgboost.pkl', 'rb'))
except:
    pass

MAX_LEN = 256

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
        #print("Bert:", np.argmax(a))
    except:
        a = np.zeros((3,))
    print("Bert:", a)
    try:
        b = svm_model.predict_proba([sent])[0]
        #print("SVM:", np.argmax(b))
    except:
        b = np.zeros((3,))
    print("SVM:", b)
    try:
        c = sentiment(sent)
        print(c,type(c))
        #print("Underthesea:",c)
    except:
        c = np.zeros((3,))
    else:
        if c == 'negative':
            c =  np.zeros((3,))
            c[2] = 1
        elif c == 'positive':
            c =  np.zeros((3,))
            c[0] = 1
    print("Underthesea:", c)
    try:
        d = xgboost_model.predict_proba([sent])
        #print("XGBoost:", np.argmax(d))
    except:
        d = np.zeros((3,))
    print("XGBoost:",d)
    final = (a + b  +d)*0.5+c
    print("Combine:",final)
    print(np.argmax(final))
    
    sent = input("Nhap cau:")
