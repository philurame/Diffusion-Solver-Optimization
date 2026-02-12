import pickle, os, torch
from torch.utils.data import Dataset

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class TrainDataset(Dataset):
  def __init__(self, split, config, device, dtype=torch.float32):
    assert split in ('train', 'val')

    with open(os.path.join(DATA, f"{config.dataset.name}.pkl"), 'rb') as f: 
      data = pickle.load(f)[split]

    size = config.dataset[f'{split}_size']
    self.data = {}
    self.data['prompts'] = data['prompts'][:size]
    self.data['noise']   = data['noise'][:size]
    self.data['imgs']    = None if data['imgs'] is None else data['imgs'][:size]
    self.data['latents'] = None if data.get('latents', None) is None else data['latents'][:size]

    # preprocess imgs to be in [0,1]
    if data['imgs'] is not None:
      if self.data['imgs'].dtype == torch.uint8:
        self.data['imgs'] = self.data['imgs'].float()/255
      if self.data['imgs'].min() < -1.1: # [-1, 1]
        self.data['imgs'] = self.data['imgs']/2 + 0.5
      assert self.data['imgs'].max() < 2

    self.device = device
    self.dtype = dtype


    self.shuffle = config.dataset.get('shuffle', False) and split == 'train'
    self._order = torch.arange(len(self))
    self._g = torch.Generator(device='cpu').manual_seed(0)
  
  def _shuffle(self):
    self._order = torch.randperm(len(self), generator=self._g)

  def __len__(self):
    return len(self.data['prompts'])

  def __getitem__(self, idx):
    # handle slice
    if isinstance(idx, slice):
      start, stop, step = idx.indices(len(self))

      # reshuffle when a new epoch begins
      if self.shuffle and start == 0:
        self._shuffle()
      
      take = self._order[start:stop:step]
      
      prompts = [self.data['prompts'][i] for i in take]
      noise   = self.data['noise'][take].to(self.device, dtype=self.dtype)
      latents = [None]*len(prompts) if self.data['latents'] is None else self.data['latents'][take].to(self.device, dtype=self.dtype)
      imgs    = [None]*len(prompts) if self.data['imgs'] is None else self.data['imgs'][take].to(self.device, dtype=self.dtype)

      return {'prompts': prompts, 'noise': noise, 'latents': latents, 'imgs': imgs}

    # integer index
    if idx < 0: idx += len(self)
    if idx < 0 or idx >= len(self): raise IndexError('Index out of range')
    idx = self._order[idx].item() if self.shuffle else idx

    prompts = self.data['prompts'][idx]
    noise   = self.data['noise'][idx].to(self.device, dtype=self.dtype)
    latents = None if self.data['latents'] is None else self.data['latents'][idx].to(self.device, dtype=self.dtype)
    imgs    = None if self.data['imgs'] is None else self.data['imgs'][idx].to(self.device, dtype=self.dtype)
    return {'prompts': prompts, 'noise': noise, 'latents': latents, 'imgs': imgs}