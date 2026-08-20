import torch.nn as nn
import torch
import lightning as pl
from transformers import get_scheduler,AutoModel

from torchmetrics.functional import classification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralClassifier(nn.Module):
    def __init__(self,model_name,num_labels,
                 full_finetune=False):
        super().__init__()
        self.backbone_model=AutoModel.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map='cuda')
        if(full_finetune==False):
            for param in self.backbone_model.parameters():
                param.requires_grad=False

        self.linear_layer = nn.Linear(768,num_labels,dtype=torch.bfloat16)
    def forward(self,real_input):
        out = self.backbone_model(input_ids=real_input["input_ids"].to(device),
                                         attention_mask=real_input["attention_mask"].to(device))
        last_hidden_state = out.last_hidden_state
        CLS_embeddings  = last_hidden_state[:,0,:]
        raw_logits = self.linear_layer(CLS_embeddings)
        return raw_logits

class BERTmodule(pl.LightningModule):
    def __init__(self,model_name,tokenizer,num_labels,sch_name,warmup_steps,total_steps,lr,
                 full_finetune=False):
        super().__init__()
        self.save_hyperparameters()
        self.neural_classifier = NeuralClassifier(model_name,num_labels,
                                                  full_finetune)
        self.neural_classifier.train()
        self.tokenizer = tokenizer
        self.sch_name = sch_name
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.automatic_optimization = False
        self.lr=lr
        self.loss_fn = nn.CrossEntropyLoss()
    def compute_loss(self,batch):
        labels = batch[1].to(device)
        titles = batch[0]
        real_input = self.tokenizer(titles,padding="longest",truncation=True,max_length=128,return_tensors="pt")
        raw_logits = self.neural_classifier(real_input)
        loss_value = self.loss_fn(raw_logits,labels)
        return raw_logits,loss_value
    def training_step(self,batch,batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        _,loss_value = self.compute_loss(batch)
        self.log("Train_loss",loss_value,on_epoch=True,on_step=False,prog_bar=True)
        self.manual_backward(loss_value)
        opt.step()
        sch.step()
        return loss_value
    def validation_step(self,batch,batch_idx):
        out_logits,loss_value = self.compute_loss(batch)
        y_pred = torch.argmax(out_logits,dim=1)
        labels = batch[1]
        self.log("Validation_loss",loss_value,on_epoch=True,on_step=False,prog_bar=True)
        self.log("Validation_Macro_F1",classification.multiclass_f1_score(y_pred,labels,num_classes=2),on_step=False,prog_bar=True,on_epoch=True)
        self.log("Validation_Satire_F1",classification.binary_f1_score(y_pred,labels),on_step=False,prog_bar=True,on_epoch=True)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),lr=self.lr)
        sch = get_scheduler(self.sch_name,opt,num_warmup_steps=self.warmup_steps,num_training_steps=self.total_steps)
        return [opt],[sch]