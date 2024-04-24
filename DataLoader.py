from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
from PIL import Image 
import torch 
from torch.utils.data import Dataset
class TokenClassificationDataset(Dataset):
 
    def __init__(self, Dataframe, processor):
        self.Dataframe = Dataframe
        self.processor = processor
 
    def __len__(self):
        return len(self.Dataframe)
 
    def __getitem__(self, item):

        # extracting words list, boxes list and labels (for TokenClassificatiion Task) list        
        words = self.Dataframe['words'][item]
        boxes = self.Dataframe['nboxes'][item]
        labels = self.Dataframe['ner_tags'][item]
        image = self.Dataframe['image'][item]

        # processing the data    
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels = labels,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            bbox=encoding["bbox"].flatten(end_dim = 1),
            pixel_values=encoding["pixel_values"].flatten(end_dim = 1),
            labels = encoding['labels'].flatten()
        )



    
