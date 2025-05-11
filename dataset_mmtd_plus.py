
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from langdetect import detect
from transformers import AutoTokenizer, AutoImageProcessor

class EmailDatasetMMTDPlus(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer_name, image_model_name, max_length=512):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.processor = AutoImageProcessor.from_pretrained(image_model_name)
        self.max_length = max_length
        self.lang2id = {}

    def get_lang_id(self, text):
        try:
            lang = detect(text)
            if lang not in self.lang2id:
                self.lang2id[lang] = len(self.lang2id)
            return self.lang2id[lang]
        except:
            return 0  # fallback if detection fails

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = int(row['label'])

        # Tokenize text
        tokens = self.tokenizer(text,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_length,
                                return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        attn_mask = tokens['attention_mask'].squeeze(0)

        # Process image
        img_path = os.path.join(self.image_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

        # Detect language ID
        lang_id = torch.tensor(self.get_lang_id(text), dtype=torch.long)

        return input_ids, attn_mask, img_tensor, lang_id, torch.tensor(label)
