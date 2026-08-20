import pandas as pd
import argparse
from bert_models import *
import spacy
from transformers import AutoTokenizer
import torch
import random
from copy import deepcopy
import torch.nn as nn
import sys
from sklearn.metrics import precision_recall_fscore_support
from aim import Run


print("LOADED Modules")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ro_nlp_spacy = spacy.load("ro_core_news_lg")
total_labels = list(ro_nlp_spacy.get_pipe("ner").labels)
skip_labels = ["NUMERIC_VALUE","DATETIME","ORDINAL","QUANTITY"]
added_tokens=[label for label in total_labels if label not in skip_labels]
added_tokens=list(map(lambda tok:"["+tok+"]",added_tokens))

train_satire = pd.read_csv("../data/train_data.csv")
tr_title_col = "proc_title" if "proc_title" in train_satire.columns else ("new_title" if "new_title" in train_satire.columns else "title")
train_satire.rename(columns={tr_title_col:"text","satiric":"label"},inplace=True)
train_satire.loc[:,"label"]=train_satire["label"].apply(int)

val_satire = pd.read_csv("../data/val_data.csv")
val_title_col = "proc_title" if "proc_title" in val_satire.columns else ("new_title" if "new_title" in val_satire.columns else "title")
val_satire.rename(columns={val_title_col:"text","satiric":"label"},inplace=True)
val_satire.loc[:,"label"]=val_satire["label"].apply(int)
train_roclico = pd.read_csv("../unsupervised_TTL/click/train.csv") #title,label
train_roclico.rename(columns={"title":"text"},inplace=True)
train_scitech = pd.read_csv("../unsupervised_TTL/scitechbait/train.csv") #headline,clickbait
train_scitech.rename(columns={"headline":"text","clickbait":"label"},inplace=True)
train_saroco = pd.read_csv("../unsupervised_TTL/saroco/train.csv") #title,label
train_saroco.rename(columns={"title":"text"},inplace=True)
train_saroco.dropna(inplace=True)


arg_parser= argparse.ArgumentParser()
arg_parser.add_argument("--model_type",type=str,default="bert")
arg_parser.add_argument("--train_style",type=str)
arg_parser.add_argument("--K",type=int)
parsed_args = arg_parser.parse_args()
model_type = parsed_args.model_type
train_style = parsed_args.train_style
K = parsed_args.K

standard_name = ""
if(model_type=="bert"):
    standard_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
elif(model_type=="xlm-roberta"):
    standard_name = "xlm-roberta-base"
elif(model_type=="distilled-bert"):
    standard_name = "racai/distilbert-base-romanian-uncased"

print(standard_name)

tokenizer = AutoTokenizer.from_pretrained(standard_name,do_lower_case=True)
tokenizer.add_tokens(added_tokens)

def reset_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


#neural_classifier.train()
loss_fn = nn.CrossEntropyLoss()
META_EPOCHS=150
INNER_BATCH_SIZE=32
betas=[1e-4,1e-3,1e-2]
alphas=[0.1,0.3,0.5,0.8]
BETA_LR=betas[0]
ALPHA_LR=alphas[-1]
END_ALPHA = 1e-5


param_dict={
    "META_EPOCHS":META_EPOCHS,
    "BATCH_SIZE":INNER_BATCH_SIZE,
    "BETA_LR":BETA_LR,
    "ALPHA_LR":ALPHA_LR,
    "END_ALPHA":END_ALPHA,
    "model_type":f"{model_type}",
    "K":K
}

def train_reptile(news_category, train_style, neural_classifier = None):
    if train_style == "separate":
        reset_seed(42)
        neural_classifier = NeuralClassifier(standard_name,len(tokenizer),num_labels=2,full_finetune=True)
        neural_classifier = neural_classifier.to(device) 
    
    aim_run = Run(experiment=f"{model_type}_{news_category}_REPTILE_{train_style}")
    aim_run["hparams"]=param_dict
    tasks_dfs = [train_satire[train_satire["category"]==news_category],train_roclico,train_saroco,train_scitech]
    validation_df = val_satire[val_satire["category"]==news_category]
    alpha_copy = param_dict["ALPHA_LR"]
    alpha_update_step = (param_dict["END_ALPHA"]-param_dict["ALPHA_LR"])/param_dict["META_EPOCHS"]
    prev_val_loss_value = None

    for epoch_no in range(META_EPOCHS):
        gradient_deltas=[]
        train_loss=0
        neural_classifier.train()
        for task_df in random.sample(tasks_dfs,k=len(tasks_dfs)-1):
            task_update = deepcopy(neural_classifier)
            df_titles = task_df.loc[:,"text"].to_list()
            df_labels = task_df.loc[:,"label"].to_list()
            pairs = list(zip(df_titles,df_labels))
            chosen_sample = random.sample(pairs,k=INNER_BATCH_SIZE*K)
            opt_inner = torch.optim.Adam(task_update.parameters(),lr=BETA_LR)
            task_loss=0
            for j in range(0,len(chosen_sample),INNER_BATCH_SIZE):
                opt_inner.zero_grad()
                batch = chosen_sample[j:j+INNER_BATCH_SIZE]
                titles=[B[0] for B in batch]
                labels=[int(B[1]) for B in batch]
                real_input=tokenizer(titles,padding='longest',max_length=128,truncation=True,return_tensors='pt')
                out_logits = task_update(real_input)
                inner_loss_value = loss_fn(out_logits,torch.tensor(labels,dtype=torch.int64).to(device))
                task_loss+=inner_loss_value.item()
                inner_loss_value.backward()
                opt_inner.step()
            train_loss+=task_loss/K
            task_gradient_update={}
            for param_name,param in task_update.named_parameters():
                task_gradient_update[param_name]=param-neural_classifier.state_dict()[param_name]
            gradient_deltas.append(task_gradient_update)
        N = len(gradient_deltas)
        aim_run.track(train_loss/N,epoch=epoch_no,name="Train_loss_meta_learning")
        #print(f"Epoch {epoch_no+1} train loss is {train_loss/3}",file=sys.stderr)
        with torch.no_grad():
            for param_name,param in neural_classifier.named_parameters():
                param_delta=torch.zeros_like(param).to(device)
                for i in range(N):
                    param_delta+=alpha_copy*gradient_deltas[i][param_name]
                param.add_(param_delta/N)
        alpha_copy+=alpha_update_step
        #########VALIDATION#######
        neural_classifier.eval()
        val_titles = validation_df.loc[:,"text"]
        val_labels = validation_df.loc[:,"label"]
        pairs = list(zip(val_titles,val_labels))
        val_loss_value=0
        steps = 0
        true_labels=[]
        predicted_labels=[]
        for j in range(0,len(pairs),INNER_BATCH_SIZE):
            steps+=1
            batch = pairs[j:j+INNER_BATCH_SIZE]
            titles=[B[0] for B in batch]
            labels=[int(B[1]) for B in batch]
            true_labels+=labels
            real_input=tokenizer(titles,padding='longest',max_length=128,truncation=True,return_tensors='pt')
            with torch.no_grad():
                out_logits = neural_classifier(real_input)
                y_pred = torch.argmax(torch.softmax(out_logits.detach().cpu(),dim=1),dim=1)
                predicted_labels+=y_pred.tolist()
                loss_value = loss_fn(out_logits,torch.tensor(labels,dtype=torch.int64).to(device))
                val_loss_value+=loss_value.item()
        #print(len(y_pred),len(true_labels))
        prec,recall,macro_f1,_ = precision_recall_fscore_support(true_labels,y_pred=predicted_labels,average="macro")
        _,_,sep_f1_score,_ = precision_recall_fscore_support(true_labels,y_pred=predicted_labels,average=None)
        aim_run.track(val_loss_value/steps,epoch=epoch_no,name="Validation_loss_meta_learning")
        aim_run.track(macro_f1,epoch=epoch_no,name="Validation_macro_F1")
        aim_run.track(sep_f1_score[1],epoch=epoch_no,name="Validation_satire_F1") 
        print(f"Epoch {epoch_no+1} val loss is {val_loss_value/steps} and has macro_f1={macro_f1} and has satire F1={sep_f1_score[1]}",file=sys.stderr)
        if prev_val_loss_value == None:
            prev_val_loss_value = val_loss_value
            torch.save(neural_classifier.state_dict(),f"backup/{train_style}_meta_learning_{model_type}_{news_category}.pt")
        elif val_loss_value < prev_val_loss_value:
            prev_val_loss_value = val_loss_value
            torch.save(neural_classifier.state_dict(),f"backup/{train_style}_meta_learning_{model_type}_{news_category}.pt")

        
if train_style == "sequential":
    reset_seed(42)
    neural_classifier = NeuralClassifier(standard_name,len(tokenizer),num_labels=2,full_finetune=True)
    neural_classifier = neural_classifier.to(device)
    for category in ["social","politic","sport"]:
        train_reptile(category, train_style, neural_classifier)
elif train_style == "separate":
    for category in ["social","politic","sport"]:
        train_reptile(category, train_style)