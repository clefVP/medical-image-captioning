import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from torch.nn.utils.rnn import pad_sequence

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        
        self.transform = transform

        
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        if len(self.captions[i])>79:
            caption = torch.LongTensor(self.captions[i][:79])
        else:
            caption = torch.LongTensor(self.captions[i])
        
            b = torch.zeros(79-caption.size()[0])
            caption = torch.cat((caption,b))
            caption = caption.type(torch.LongTensor)
        
        if self.caplens[i]>79:
            caplen = torch.LongTensor([79])
        else:
            caplen = torch.LongTensor([self.caplens[i]])
        
        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            
            all_captions = torch.LongTensor(
                caption)
            
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
    
