from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_SAINT_GALL = "/data/users/amolina/OCR/SaintGall"

class SaintGallDataset(GenericDataset):
    name = 'saint_gall_dataset'
    def __init__(self, base_folder = DEFAULT_SAINT_GALL, split: ['train', 'test', 'valid'] = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()

        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms
        self.data = []

        valid_pages = [x.strip()
                            for x in open(
                                os.path.join(base_folder, 'sets', split + '.txt')
                            )
                        ]

        for line in ([x.strip() for x in open(
                                        os.path.join(
                                            base_folder, 'ground_truth', 'transcription.txt'
                                        ), 'r').readlines()]):
            
            id_line, _, transcription = line.split()
            id_page = '-'.join(
                id_line.split('-')[:2]
            )

            file_path = os.path.join(
                base_folder, 'data', 'line_images_normalized', id_line + '.png'
            )
            if not id_page in valid_pages: continue
            transcription_tokens = transcription\
                                    .replace('|pt|', '|.|')\
                                    .replace('|et|', '|&|')\
                                    .split('|')

            self.data.append(
                {
                    'image_path': file_path,
                    'transcription': ' '.join(transcription_tokens)
                }
            )
            

                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        metadata = self.data[idx]
        
        
        image = Image.open(
                           
                            metadata['image_path']
    
                           ).convert('RGB')

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



        
