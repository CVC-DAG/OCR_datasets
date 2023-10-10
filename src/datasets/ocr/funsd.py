from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_FUNSD = "/data/users/amolina/OCR/FUNSD"

class FUNSDDataset(GenericDataset):
    name = 'funsd_dataset'
    
    def __init__(self, base_folder = DEFAULT_FUNSD, split: ['train', 'test'] = 'train', patch_width = 16, image_height = 128, transformations = lambda x: x) -> None:
        super().__init__()
        
        split_rename = 'testing_data' if split=='test' else 'training_data'

        self.split = split
        self.patch_width = patch_width
        self.image_height = image_height
        self.transforms = transformations
        
        folder = os.path.join(base_folder, split_rename)
        self.base_images = os.path.join(folder, 'images')
        self.base_json = os.path.join(folder, 'annotations')
        
        
        annotations = os.listdir(
            self.base_json
        )
        sample_ids = {
            ann.replace('.json', ''): {'json_file': os.path.join(self.base_json, ann)} for ann in annotations
        }
        
        images = os.listdir(
            self.base_images
        )
        for image in images:
            name = image.split('.')[0]
            sample_ids[name]['image_file'] = image
        
        self.samples = []
        for id_ann in sample_ids:
            
            annotation_data = json.load(
                open(sample_ids[id_ann]['json_file'], 'r')
            )
            for form in annotation_data['form']:
                for word in form['words']:
                    self.samples.append(
                        {
                            'image_path': sample_ids[id_ann]['image_file'],
                            'bbx': word["box"],
                            "transcription": word['text']
                            
                        }
                    )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        metadata = self.samples[idx]
        x,y,w,h = metadata['bbx']

        image = Image.open(
                           
                           os.path.join(self.base_images, metadata['image_path'])
                           
                           ).crop((x, y, w, h)).convert('RGB')
        
        original_width, _ = image.size
        new_width = original_width + (original_width % self.patch_width)
        
        image_resized = image.resize((new_width, self.image_height))
        input_tensor = self.transforms(image_resized)
        
        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": metadata['transcription'],
            'dataset': self.name,
            'split': self.split
        }
            