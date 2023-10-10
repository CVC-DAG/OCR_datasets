from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_TEXTOCR = "/data/users/amolina/OCR/TextOCR"

class TextOCRDataset(GenericDataset):

    name = 'text_ocr_dataset'
    def __init__(self, base_folder = DEFAULT_TEXTOCR, split: ['train', 'val'] = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms
        self.data = []

        annotations = json.load(
            open(
                os.path.join(base_folder, f"TextOCR_0.1_{split}.json"), 'r'
            )
        )
        valid_images = [
                        
                        {
                            'path': os.path.join(
                                                    base_folder, annotations['imgs'][img]['file_name'].replace('train/', 'images/')
                                                 ),
                            'annots': [str(i) for i in annotations['imgToAnns'][img]],
                        } 
                            
                            for img in annotations['imgs']
                        ]

        for image in valid_images:
            for ann in image['annots']:
                
                annotation = annotations['anns'][ann]
                if annotation['utf8_string'] == '.': continue # Unreadable characters
                self.data.append({
                    'image_path': image['path'],
                    'bbx': [int(x) for x in annotation['bbox']],
                    'transcription': annotation['utf8_string'],
                })
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        metadata = self.data[idx]
        x,y,w,h = metadata['bbx']
        
        
        image = Image.open(
                           
                            metadata['image_path']
                           
                           ).crop((x, y, x+w, y+h)).convert('RGB')
        
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