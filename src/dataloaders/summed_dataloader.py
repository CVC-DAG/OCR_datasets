import numpy as np
from PIL import Image
import torch

class GenericDataset:

    def add(self, dataset):
        return SummedDataset(self, dataset)

    def resize_image(self, image):

        original_width, original_height = image.size
        
        original_height = max(original_height, 1)
        original_width = max(original_width, 1)
        
        scale = self.image_height / original_height
        
        resized_width = int(round(scale * original_width, 0))
        new_width = resized_width + (self.patch_width - (resized_width % self.patch_width))  # Adjusted this line
        
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
        self.visual_padding_token = torch.zeros((3, self.image_height, self.patch_width))
    
    def collate(self, batch):

        max_patches = max([x['input_tensor'].shape[2] // self.patch_width for x in batch])
        max_tokens = max([3 + len(x['tokens']) for x in batch])
        
        visual_tokens = []
        text_tokens = []
        raw_texts = []
        sources = []
        resized_images = []
        
         
        for item in batch:
            
            image, text_token, raw_text, split, dataset, original_image = item['input_tensor'], item['tokens'], item['annotation'], item['split'], item['dataset'], item['resized_image']
            
            resized_images.append(original_image)
            sources.append(f"{split}_ {dataset}")
            raw_texts.append(raw_text)
            
            text_tokens.append(torch.from_numpy(
                self.character_tokenizer(
                    text_token + [self.character_tokenizer.padding_token] * (max_tokens - len(text_token))
                )
            ))


            patches = list(image.chunk(image.shape[2] // self.patch_width, dim=-1))
            
            patches = patches + [self.visual_padding_token] * (max_patches - len(patches))
            
            visual_tokens.append(
                torch.stack(
                    patches
                )
            )
    
        return {
                'input_visual_seq': torch.stack(visual_tokens),
                'labels': torch.stack(text_tokens),
                'raw_text_gt': raw_texts,
                'sources': sources,
                'original_images': resized_images
                }


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