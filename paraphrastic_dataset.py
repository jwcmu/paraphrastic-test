import h5py
import torch
import numpy as np


class ParaphrasticDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, data_key):
        super(ParaphrasticDataset, self).__init__()
        self.h5_file = h5py.File(h5_file, "r")
        self.data = self.h5_file[data_key]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __del__(self):
        self.h5_file.close()

def paraphrastic_collate_fn(data):
  s1 = [i[0] for i in data]
  s1_lengths = [len(i) for i in s1]
  pad_width = [max(s1_lengths) - i for i in s1_lengths]
  s1 = [np.pad(i, (0,j), mode="constant", constant_values=0) for i,j in zip(s1, pad_width)]
  s1 = np.stack(s1)
  s1 = torch.tensor(np.array(s1), dtype=torch.int32)

  s2 = [i[1] for i in data]
  s2_lengths = [len(i) for i in s2]
  pad_width = [max(s2_lengths) - i for i in s2_lengths]
  s2 = [np.pad(i, (0,j), mode="constant", constant_values=0) for i,j in zip(s2, pad_width)]
  s2 = np.stack(s2)
  s2 = torch.tensor(np.array(s2), dtype=torch.int32)
  return s1, s2, torch.tensor(s1_lengths, dtype=torch.int32), torch.tensor(s2_lengths, dtype=torch.int32)
