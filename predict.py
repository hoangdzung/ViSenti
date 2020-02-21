import pickle
import torch 
from scipy.special import softmax 
from preprocessing import * 
from transformers import *
from underthesea import sentiment

MAX_LEN = 256 

def bert_predict(sent, tokenizer=None):
    encoded_sent = tokenizer.encode(sent ,add_special_tokens = True)
    if len(encoded_sent) > MAX_LEN:
        encoded_sent = encoded_sent[:MAX_LEN-1]+ encoded_sent[-2:-1]                                                  
    outputs = bert_model(torch.tensor([encoded_sent]))
    return softmax(outputs[0].detach().cpu().numpy())

def predict(sent, bert_model=None, tokenizer=None,xgboost_model=None, svm_model=None):
    """
        0: NEG, 1: NEU, 2:POS

    """
    try:
        bert_out = bert_predict(sent, tokenizer)[0]
    except:
        bert_out = np.zeros((3,))

    try:
        xgb_out = xgboost_model.predict_proba([sent])[0]
    except:
        xgb_out = np.zeros((3,))

    try:
        svm_out = svm_model.predict_proba([sent])[0]
    except:
        svm_out = np.zeros((3,))

    try:
        underthesea_out = sentiment(sent)
    except:
        underthesea_out = np.zeros((3,))
    else:
        if underthesea_out == 'negative':
            underthesea_out =  np.array([1,0,0])
        elif underthesea_out == 'positive':
            underthesea_out =  np.array([0,0,1])

    final = (bert_out + xgb_out  +svm_out)*0.5 + underthesea_out
    label = np.argmax(final)
    assert label in range(3)
    if label == 0:
        return "negative"
    elif label == 1:
        return "neutral"
    elif label == 2:
        return "positive"

if __name__ == '__main__':
    try:
        bert_model = torch.load('bert.pt')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
    except:
        bert_model = None 
        tokenizer = None

    try:
        svm_model = pickle.load(open('svm.pkl', 'rb'))
    except:
        svm_model = None

    try:
        xgboost_model = pickle.load(open('model_xgboost.pkl', 'rb'))
    except:
        xgboost_model = None

    sent = input('Sentence: ')
    while(len(sent) > 0):
        print(predict(sent, bert_model=bert_model, tokenizer=tokenizer,xgboost_model=xgboost_model, svm_model=svm_model))
        print("Enter to quit")
        sent = input('Sentence: ')
