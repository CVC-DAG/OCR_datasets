from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_PARZIVAL = "/data/users/amolina/OCR/Parzival"

class ParzivalDataset(GenericDataset):
    name = 'parzival_dataset'
    def __init__(self, base_folder = DEFAULT_PARZIVAL, split: ['train', 'test', 'val'] = 'train', mode = 'word', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        
        self.split = split
        self.patch_width = patch_width
        self.image_height = image_height
        self.mode = mode
        self.transforms = transforms
        
        valid_lines = [x.strip() for x in
                            open(
                                os.path.join(base_folder, 'sets1' if mode =='line' else 'sets2', split + '.txt'), 'r'
                            ).readlines()
                        ]
        self.images_path = os.path.join(
            base_folder, 'data', f'{mode}_images_normalized'
        )
        
        groundtruth = [x.strip() for x in
                            open(
                                os.path.join(base_folder, 'ground_truth', 'transcription.txt' if mode == 'line' else 'word_labels.txt'), 'r'
                            ).readlines()
                        ]
        
        self.data = []
        for gt in groundtruth:

            file_id, transcription = gt.split()
            if file_id in valid_lines:

                img_path = os.path.join(base_folder, "data", f"{mode}_images_normalized", file_id + '.png')
                line_transcription = transcription\
                .replace("pt", ".").replace('|', '- -') # TODO: Solve strange characters

                self.data.append({
                    'image_path': img_path,
                    'transcription': line_transcription,
                    'tokens':  [char if char != 'eq' else '-' for char in line_transcription.split('-')]
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        metadata = self.data[idx]

        image = Image.open(metadata['image_path']).convert('RGB')
                
        image_resized = self.resize_image(image)

        input_tensor = self.transforms(image_resized)
        
        annotation = metadata['transcription']
        
        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": annotation,
            'dataset': self.name,
            'split': f"{self.mode}_{self.split}",
            'path': metadata['image_path'],
            'tokens': metadata['tokens']
        }
                        


