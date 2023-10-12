import torch
import numpy as np
import torch.nn as nn
from torch import autograd

import sys
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset

from monai.transforms import Compose, Randomizable
from monai.utils import MAX_SEED, get_seed
from monai.data import DataLoader, Dataset,ZipDataset
from monai.data import (
    Dataset,ArrayDataset)
from torch.utils.data import Dataset as _TorchDataset
from monai.data import DataLoader

from monai.transforms import (
    Compose,
    ToTensor
)
from model import *



class ArrayDataset(Randomizable, _TorchDataset):
      
  def __init__(
        self,
        img: Sequence,
        img_transform: Optional[Callable] = None,
        seg: Optional[Sequence] = None,
        seg_transform: Optional[Callable] = None,
        labels: Optional[Sequence] = None,
        label_transform: Optional[Callable] = None,
        labels2: Optional[Sequence] = None,
        label2_transform: Optional[Callable] = None,
        labels3: Optional[Sequence] = None,
        label3_transform: Optional[Callable] = None
        
    ) -> None:
   
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform),(labels2,label2_transform),(labels3,label3_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed


  def __len__(self) -> int:
        return len(self.dataset)
        
  def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")


  def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index]
  
def create_datasets(patches, patches_mask, patches_maskmimi, patches_label,
                    patches_valid, patches_valid_mask, patches_valid_maskmimi, patches_valid_label):
    transform = Compose([ToTensor()])

    DS = ArrayDataset(patches, transform, patches_mask, transform, patches_maskmimi, transform, patches_label, transform)
    DSV = ArrayDataset(patches_valid, transform, patches_valid_mask, transform, patches_valid_maskmimi, transform, patches_valid_label, transform)
    
    return DS, DSV

def get_data_train(DS,bs):
    train_loader = DataLoader(DS, batch_size=bs,num_workers=2, shuffle=True)
    return train_loader

def get_data_valid(DSV,bsv):
    valid_loader = DataLoader(DSV, batch_size=bsv,num_workers=2, shuffle=False)
    return valid_loader

def laplacian2(input):
    input = input.squeeze(0)
    LP = torch.zeros_like(input)
    fieldft = torch.fft.fftn(input)
    LP[0,0,0] = -6.0
    LP[1,0,0] = 1.0
    LP[-1,0,0] = 1
    LP[0,1,0] =1
    LP[0,-1,0]=1
    LP[0,0,1]=1
    LP[0,0,-1]=1
    FLP = torch.fft.fftn(LP)
    
    result = torch.fft.ifftn(fieldft * FLP)
    result = torch.real(result)
    return result

def l1_loss(x, y):

    l1 = tf.reduce_mean(tf.reduce_mean(tf.abs(x - y), [1, 2, 3, 4]))

    return l1



def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = netD(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size(), device=real_data.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
