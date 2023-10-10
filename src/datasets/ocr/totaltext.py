from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_TOTALTEXT = "/data/users/amolina/OCR/TotalText"

class TotalTextDataset(GenericDataset):
    name = 'total_text_dataset'

    def __init__(self, base_folder = DEFAULT_TOTALTEXT, split: ['Train', 'Test'] = 'Train', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms
        self.data = []

        gt_data = [os.path.join(base_folder, split, file) for file in os.listdir(
            os.path.join(base_folder, split)
        ) if file.endswith('.txt')]

        image_folder = os.path.join(
            base_folder, 'Images', split
        )

        for data in gt_data:

            image_file = data.split('/')[-1].replace('.txt', '.jpg').replace('poly_gt_', '')
            image_path = os.path.join(
                image_folder, image_file
            )

            for line in [a.strip() for a in open(data, 'r').readlines()]:

                line = line\
                    .replace("x: [[", "'x': '")\
                    .replace("y: [[", "'y': '")\
                    .replace("]], ", "', ")\
                    .replace("ornt: [", "'ornt': [")\
                    .replace("transcriptions: [", "'transcriptions': [") # Mare de Déu que és això si us plau
                line_as_dict = eval("{" + line + "}")

                points = [(int(x), int(y)) for x,y in zip(line_as_dict['x'].split(), line_as_dict['y'].split())]

                box = (min(points, key = lambda x: x[0])[0], min(points, key = lambda x: x[1])[1], max(points, key = lambda x: x[0])[0], max(points, key = lambda x: x[1])[1])
                self.data.append({
                    'image_path': image_path,
                    'bbx': box,
                    'transcription': ' '.join(line_as_dict['transcriptions'])

                })
            
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
                