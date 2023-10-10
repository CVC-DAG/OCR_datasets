from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_COCOTEXT = "/data/users/amolina/OCR/COCOText"

class COCOTextDataset(GenericDataset):
    name = 'cocotext_dataset'
    def __init__(self, base_folder = DEFAULT_COCOTEXT, annots_name='cocotext.v2.json', split = 'train',
                 langs = ['english', 'not english'], legibility = ['legible', 'illgible'],
                 image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()

        # split train / val
        json_annots = json.load(
            open(os.path.join(base_folder, annots_name))
        )
        valid_images = [{
                            'path': json_annots['imgs'][img]['file_name'],
                            'annots': [str(i) for i in json_annots['imgToAnns'][img]],
                        } 
                            
                            for img in json_annots['imgs'] if json_annots['imgs'][img]['set'] == split
                        ]
        
        self.samples = {}
        total_count = 0
        for img in valid_images:
            for annot in img['annots']:
                
                annotation = json_annots['anns'][annot]
                if annotation['legibility'] in legibility and annotation['language'] in langs:
                    self.samples[total_count] = {
                        'image_path': img['path'],
                        'bbx': [int(x) for x in annotation['bbox']],
                        'transcription': annotation['utf8_string'],
                    }
                    total_count += 1

        self.base_images = os.path.join(base_folder, 'train2014')
        self.image_height = image_height
        self.patch_width  = patch_width 
        
        self.transforms = transforms
        self.split = split + '_'.join(langs + legibility)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        metadata = self.samples[idx]
        x,y,w,h = metadata['bbx']
        
        
        image = Image.open(
                           
                           os.path.join(self.base_images, metadata['image_path'])
                           
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
            
                      

        
