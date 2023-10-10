from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_SROIE = "/data/users/amolina/OCR/SROIE"

class SROIEDataset(GenericDataset):
    name = 'sroie_dataset'

    def __init__(self, base_folder = DEFAULT_SROIE, split: ['train', 'test'] = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms
        self.data = []

        ids = set(
                    [image_id.split('.')[0] for image_id in os.listdir(
                                                os.path.join(
                                                    base_folder, split
                                                )
                    )])
        
        for img_id in ids:

            image_path = os.path.join(
                base_folder, split, img_id + '.jpg'
            )

            annotations = [
                x.strip() for x in open(
                    os.path.join(base_folder, split, img_id + '.txt'), errors='ignore'
                ).readlines()
            ] # With Errors = ignore you just skip bad characters

            for annotation in annotations:

                rows = annotation.split(',')
                if len(rows) == 1: continue # empty row
                xx, xy, yy, yx, x2x, x2y, y2y, y2x, transcription = rows[:8] + [','.join(rows[8:])]
                points = [[int(y) for y in x] for x in [[xx, xy],
                                  [yy, yx],
                                  [x2x, x2y],
                                  [y2y, y2x]]]
                        

                self.data.append({
                    
                    'image_path': image_path,
                    'transcription': transcription,
                    'bbx': (min(points, key = lambda x: x[0])[0], min(points, key = lambda x: x[1])[1], max(points, key = lambda x: x[0])[0], max(points, key = lambda x: x[1])[1])

                    }
                )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        metadata = self.data[idx]
        x,y,w,h = metadata['bbx']
        
        
        image = Image.open(
                           
                            metadata['image_path']
                           
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

