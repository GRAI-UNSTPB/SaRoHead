import lightning as pl
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification, set_seed
import argparse
from bert_models import *
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
import spacy

class HeadlineDataset(Dataset):
    def __init__(self,category,df_name):
        self.df = pd.read_csv(df_name)
        if category!="all":
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
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_type",default="bert")
arg_parser.add_argument("--lr")
arg_parser.add_argument("--sch_type")
arg_parser.add_argument("--inter_task")
arg_parser.add_argument("--num_epochs")
arg_parser.add_argument("--full_finetune",action="store_true")
parsed_args = arg_parser.parse_args()
sch_name = parsed_args.sch_type
initial_lr = float(parsed_args.lr)
inter_task = parsed_args.inter_task
model_type = parsed_args.model_type
batch_size = 32
num_epochs = int(parsed_args.num_epochs)
folder_suffix={
    "bert":"BERT",
    "xlm-roberta":"XLM",
    "distilled-bert":"Distil"
}

def train_category(category_name, inter_task):


    trainData = HeadlineDataset(category_name,"../data/train_data.csv")
    valData = HeadlineDataset(category_name,"../data/val_data.csv")
    standard_name = ""
    if(model_type=="bert"):
        standard_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
    elif(model_type=="xlm-roberta"):
        standard_name = "xlm-roberta-base"
    elif(model_type=="distilled-bert"):
        standard_name = "racai/distilbert-base-romanian-uncased"
    model_path = standard_name if inter_task=="standard" else f'../unsupervised_TTL/{inter_task}/backup_{inter_task}_{folder_suffix[model_type]}'
    pl.seed_everything(42)
    trainLoader = DataLoader(trainData,shuffle=True,batch_size=batch_size,collate_fn=collate_fn)
    valLoader = DataLoader(valData,shuffle=False,batch_size=batch_size,collate_fn=collate_fn)
    total_steps = num_epochs * len(trainLoader)
    tokenizer = AutoTokenizer.from_pretrained(standard_name)
    full_model = BERTmodule(model_path,tokenizer,num_labels=2,sch_name=sch_name,warmup_steps=0,total_steps=total_steps,lr=initial_lr,
                            full_finetune=parsed_args.full_finetune)
    additional_suffix = inter_task
    model_save = pl.pytorch.callbacks.ModelCheckpoint(dirpath=f"{category_name}_satire_from_{model_type}_{additional_suffix}_finetune_{parsed_args.full_finetune}",
                                                        filename=f"{model_type}_CKPT",
                                                        enable_version_counter=False)
    
    aim_logger=AimLogger(experiment=f"train_{model_type}_{category_name}_{additional_suffix}")
    aim_logger.log_hyperparams({
        'lr':initial_lr,
        'batch_size':batch_size,
        'lr_sched':sch_name,
        'num_epochs':num_epochs,
        'model_type':model_type
    })
    csv_logger = CSVLogger(f"{model_type}_{additional_suffix}_classif",f"{model_type}_logs")

    trainer = pl.Trainer(max_epochs=num_epochs,precision="bf16",logger=[aim_logger,csv_logger],callbacks=[model_save])
    trainer.fit(full_model,trainLoader,valLoader)
        

if __name__=='__main__':
    for category in ["social","politic","sport"]:
        print("#"*100)
        print(f"TRAINING {category}")
        train_category(category, inter_task)