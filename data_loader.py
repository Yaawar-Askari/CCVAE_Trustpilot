import random
import numpy as np
from itertools import cycle
from utils import imshow_grid, transform_config
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TrustpilotDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, dtype={'rating': int})
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # To do: get the tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data['comment'][index]
        rating = self.data['rating'][index]

        # Tokenize the comment and convert to tensor
        tokenized_comment = self.tokenizer(comment, padding='max_length', max_length=100, truncation=True, return_tensors='pt')['input_ids'].squeeze(0)

        return tokenized_comment, rating

if __name__ == '__main__':
    trustpilot_dataset = TrustpilotDataset('data/trustpilot.csv')
    loader = cycle(DataLoader(trustpilot_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True))

    comment_batch, rating_batch = next(loader)
    print(comment_batch)
    print(rating_batch)
