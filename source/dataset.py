import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from midi_utils import event_list_to_tokens

class MaestroDataset(Dataset):
    def __init__(self, dataset_hparams):
        super().__init__()
        self.hp = dataset_hparams
        self.files = [os.path.join(self.hp.root_dir, file) for file in os.listdir(self.hp.root_dir) if 'npy' in file]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        event_list = np.load(file, allow_pickle=True)
        tokens = event_list_to_tokens(event_list, self.hp)
        if len(tokens) < self.hp.token_length:
            start_index = 0
        else:
            start_index = np.random.randint(0, len(tokens)-self.hp.token_length)
        tokens = np.array(tokens[start_index:start_index+self.hp.token_length])
        tokens_padded = np.zeros(self.hp.token_length, dtype=np.int)
        tokens_padded[:len(tokens)] = tokens
        return np.array(tokens_padded)