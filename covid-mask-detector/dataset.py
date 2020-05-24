""" Dataset module
"""
import cv2
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


class MaskDataset(Dataset):
    """ Masked faces dataset
        0 = 'no mask'
        1 = 'mask'
    """
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(), # [0, 1]
        ])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        return {
            'image': self.transformations(cv2.imread(row['image'])),
            'mask': tensor([row['mask']], dtype=long), # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)
