import numpy as np
from PIL import Image

class GenericDataset:

    def add(self, dataset):
        return SummedDataset(self, dataset)

    def resize_image(self, image):

        original_width, original_height = image.size
        scale = self.image_height / original_height
        
        resized_width = int(round(scale * original_width, 0))
        new_width = resized_width + (resized_width % self.patch_width)
        
        return image.resize((new_width, self.image_height))


    def __add__(self, dataset):
        return self.add(dataset)

class SummedDataset(GenericDataset):
    def __init__(self, dataset_left, dataset_right) -> None:
        self.left = dataset_left
        self.right = dataset_right

    def __len__(self):
        return len(self.left) + len(self.right)

    def __getitem__(self, idx):

        if idx > (len(self.left) - 1):
            idx_corrected = idx % len(self.left)
            return self.right[idx_corrected]
        
        return self.left[idx]
        
class DummyDataset(GenericDataset):

    def __init__(self, number = 30, name = 'dummy_v1', split = 'test') -> None:

        self.samples = list(range(number))
        self.name = name
        self.split = split
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'dataset_name': self.name,
            'split': self.split,
            'image': self.samples[idx]
        }

class CollateFNs:

    def __init__(self, patch_width, image_height, character_tokenizer) -> None:
        self.patch_width = patch_width
        self.image_height = image_height
        self.character_tokenizer = character_tokenizer
    
    def collate(self, batch):
        ## ADD 0 - PADDING SO WE CAN HORIZONTALY TOKENIZE THE SEQUENCES ##
        pass

if __name__ == '__main__':

    dataset_0_3 = DummyDataset(3, name = 'dataset_1', split = 'test')
    dataset_3_5 = DummyDataset(2, name = 'dataset_2', split = 'test')
    dataset_5_10 = DummyDataset(5, name = 'dataset_3', split = 'val')

    dataset_0_5 = dataset_0_3 + dataset_3_5
    dataset_0_10 = dataset_0_5 + dataset_5_10

    print(dataset_0_10[0]) # Prints dataset_1: 0
    print(dataset_0_10[3]) # Prints dataset_2: 0
    print(dataset_0_10[5]) # Prints dataset_3: 0

    print(dataset_0_10[2]) # Prints dataset_1: 2
    print(dataset_0_10[4]) # Prints dataset_2: 1
    print(dataset_0_10[9]) # Prints dataset_3: 4