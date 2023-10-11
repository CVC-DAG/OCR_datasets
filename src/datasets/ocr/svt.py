from PIL import Image
import os
import json
from bs4 import BeautifulSoup as bs

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_SVT = "/data/users/amolina/OCR/SVT"

class SVTDataset(GenericDataset):
    name = 'svt_dataset'

    def __init__(self, base_folder = DEFAULT_SVT, split: ['train', 'test'] = 'test', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms
        self.data = []

        xml_soup = bs(
            ''.join(
                open(

                    os.path.join(base_folder, split + '.xml')

                )), features = 'lxml'
        )

        images = xml_soup.find_all('image')
        for image in images:
            
            image_path = os.path.join(base_folder, image.find('imagename').text)

            for rect in image.find_all('taggedrectangle'):
                x, y, h, w = rect['x'], rect['y'], rect['width'], rect['height']
                self.data.append(
                    {
                        'image_path': image_path,
                        'bbx': (int(a) for a in (x, y, h, w)),
                        'transcription': rect.find('tag').text
                    }
                )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        metadata = self.data[idx]
        x,y,w,h = metadata['bbx']
        
        
        image = Image.open(
                           
                            metadata['image_path']
                           
                           ).crop((x, y, x+w, y+h)).convert('RGB')
        
        image_resized = self.resize_image(image)

        input_tensor = self.transforms(image_resized)
        
        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": metadata['transcription'],
            'dataset': self.name,
            'split': self.split,
            'tokens': [char for char in metadata['transcription']]

        }