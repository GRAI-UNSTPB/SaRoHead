import lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import argparse
from bert_models import *
from aim.pytorch_lightning import AimLogger
from torchmetrics import functional
import spacy
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import numpy as np
class HeadlineDataset(Dataset):
    def __init__(self,category,df_name):
        self.df = pd.read_csv(df_name)
        self.df = self.df[self.df["category"]==category]
        title_col = "proc_title" if "proc_title" in self.df.columns else ("new_title" if "new_title" in self.df.columns else "title")
        self.titles = self.df[title_col]
        self.labels = self.df["satiric"].to_numpy()
    def __getitem__(self,idx):
        return self.titles.iloc[idx],self.labels[idx]
    def __len__(self):
        return self.labels.shape[0]


  
def collate_fn(batch):
    titles = []
    labels = []
    for title,label in batch:
        titles.append(title)
        labels.append(label)
    labels = torch.tensor(labels).to(dtype=torch.int64)
    return titles,labels



ro_nlp_spacy = spacy.load("ro_core_news_lg")
total_labels = list(ro_nlp_spacy.get_pipe("ner").labels)
skip_labels = ["NUMERIC_VALUE","DATETIME","ORDINAL","QUANTITY"]
added_tokens=[label for label in total_labels if label not in skip_labels]
added_tokens=list(map(lambda tok:"["+tok+"]",added_tokens))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


standard_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
def test_category(category_name):

    pl.seed_everything(42)
    testData = HeadlineDataset(category_name,"../data/test_data.csv")
    testLoader = DataLoader(testData,shuffle=False,batch_size=32,collate_fn=collate_fn)
    valData = HeadlineDataset(category_name,"../data/val_data.csv")
    tokenizer=AutoTokenizer.from_pretrained(standard_name,do_lower_case=True)
    tokenizer.add_tokens(added_tokens)
    full_model=NeuralClassifier(standard_name,len(tokenizer),num_labels=2,full_finetune=True)
    full_model.load_state_dict(torch.load(f"backup/meta_learning_bert_{category_name}.pt",weights_only=True))
    full_model = full_model.to(device)
    full_model.eval()
    total_preds = torch.tensor([])
    total_GT = torch.tensor([])
    for batch in testLoader:
        titles = batch[0]
        tokenized_titles = tokenizer(titles,return_tensors="pt",padding='longest',max_length=128,truncation=True)
        labels = batch[1]
        total_GT=torch.concatenate((total_GT,labels))
        out_logits = full_model(tokenized_titles)
        y_pred = torch.argmax(out_logits.detach().cpu(),dim=1)
        total_preds = torch.concatenate((total_preds,y_pred))
    conf_matrix = confusion_matrix(y_true=total_GT.numpy(),y_pred=total_preds.numpy())
    fig = ConfusionMatrixDisplay(conf_matrix)
    fig.plot().figure_.savefig(f"binary_conf_matrix_meta_learning_bert_{category_name}.png")
    mispredictions=[]
    for i in range(total_preds.shape[0]):
        if(total_preds[i]!=total_GT[i]):
            mispredictions.append([testData[i][0],int(total_preds[i].item()),int(total_GT[i].item())])
    pd.DataFrame(mispredictions,columns=["title","prediction","truth"]).to_csv(f"../results_classif/best_model_results/mispredictions_meta_learning_bert_{category_name}.csv")

if __name__=="__main__":
    for category_name in ["social","politic","sport"]:
        print("#"*100)
        print("#"*100)
        test_category(category_name)
