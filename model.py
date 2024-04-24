RECEIPT_CLASSES = ['O', 'COMPANY' , 'LOCATION' , 'DATE' , 'TOTAL', 'TAX',  'CURRENCY']
import pytorch_lightning as pl
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast,LayoutLMv3Processor
from torchmetrics import Accuracy
import torch
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


class ModelModule(pl.LightningModule):
    def __init__(self, n_classes:int, weight_decay = 1e-5 , learning_rate = 5e-5, t_total = 1000, adam_epsilon = 1e-8 ):
        super().__init__()
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=n_classes
        )
        self.model.config.id2label = {k: v for k, v in enumerate(RECEIPT_CLASSES)}
        self.model.config.label2id = {v: k for k, v in enumerate(RECEIPT_CLASSES)}
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.t_total = t_total
        self.adam_epsilon = adam_epsilon


 
    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )
 
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.log("train_loss", output.loss)
        logits = output.logits
        self.log(
            "train_acc",
            self.train_accuracy(logits.permute(0,2,1), labels),
            on_step=True,
            on_epoch=True
        )
        print(f"Training accuracy :{self.train_accuracy(logits.permute(0,2,1), labels)} ")

        return output.loss
 
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.log("val_loss", output.loss)
        logits = output.logits
        self.log(
            "val_acc",
            self.val_accuracy(logits.permute(0,2,1), labels),
            on_step=False,
            on_epoch=True
        )
        print(f"Validation accuracy :{self.val_accuracy(logits.permute(0,2,1), labels)} ")
        return output.loss
 
    def configure_optimizers(self):
                # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.weight_decay,
        },
        {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total
        )

        return optimizer