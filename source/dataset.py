import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict

dims = EasyDict(interval = 100,
                velocity = 32,
                note_on = 128,
                note_off = 128,
                pedal_on = 1,
                pedal_off = 1)

offsets = EasyDict(interval = 100,
                   velocity = dims.interval,
                   note_on = dims.interval + dims.velocity,
                   note_off = dims.interval + dims.velocity + dims.note_on,
                   pedal_on = dims.interval + dims.velocity + dims.note_on + dims.note_off,
                   pedal_off = dims.interval + dims.velocity + dims.note_on + dims.note_off + dims.pedal_on)

dataset_hparams = EasyDict(root_dir = 'dataset/',
                           max_note_duration = 2, # seconds)
                           token_length = 500,
                           dims = dims,
                           offsets = offsets
                          )

def event_list_to_tokens(event_list, hp):
    tokens = []
    current_time = 0
    for event in event_list:
        interval = event['time'] - current_time
        interval_token = int(interval / hp.max_note_duration * hp.dims.interval)
        interval_token = min(interval_token, hp.dims.interval)
        tokens.append(interval_token)
        current_time = event['time']
        
        if event['type'] == 'note_on':
            tokens.append(hp.offsets.velocity + int(event['velocity'] / 128 * hp.dims.velocity))
            tokens.append(hp.offsets.note_on + event['note'])
        elif event['type'] == 'note_off':
            tokens.append(hp.offsets.note_off + event['note'])
        elif event['type'] == 'pedal_on':
            tokens.append(hp.offsets.pedal_on)
        elif event['type'] == 'pedal_off':
            tokens.append(hp.offsets.pedal_off)
    return np.array(tokens)

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