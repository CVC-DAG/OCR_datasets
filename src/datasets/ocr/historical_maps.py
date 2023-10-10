from PIL import Image
import os
import json 
from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_HIST_MAPS = "/data/users/amolina/OCR/HistoricalMaps"

class HistoricalMapsdDataset(GenericDataset):
    name = 'hist_maps'

    def __init__(self, base_folder = DEFAULT_HIST_MAPS, split: ["train", "test"] = 'train', cross_val = 'cv1', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        
        self.image_height = image_height
        self.patch_width = patch_width

        self.base_gt_folder = os.path.join(base_folder, 'GT')
        self.base_img_folder = os.path.join(base_folder, 'maps')


        valid_records = [record.strip() for record in open(os.path.join(base_folder, 'splits', f"{cross_val}_{split}.txt"), 'r').readlines()]
        samples = []
        for file in os.listdir(self.base_gt_folder):
            
            if file in valid_records:
                img_file = os.path.join(
                    self.base_img_folder,
                    file.replace('.json', '.tiff')
                )
                gt_file = os.path.join(self.base_gt_folder, file)

                for item in json.load(open(gt_file, 'r')):
                    for word in item['items']:

                        points = word['points']

                        samples.append({
                            
                            'image_path': img_file,
                            'transcription': word['text'],
                            'bbx': (min(points, key = lambda x: x[0])[0], min(points, key = lambda x: x[1])[1], max(points, key = lambda x: x[0])[0], max(points, key = lambda x: x[1])[1])

                            }
                        )
        self.data = samples
        self.split = f"{split}_{cross_val}"
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        metadata = self.data[idx]
        x,y,w,h = metadata['bbx']
        
        
        image = Image.open(
                           
                           os.path.join(self.base_img_folder, metadata['image_path'])
                           
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

