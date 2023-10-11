from PIL import Image
import os
import json 
from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_IAM = "/data/users/amolina/OCR/IAM"

class IAMDataset(GenericDataset):
    name = 'iam_dataset'

    def __init__(self, base_folder = DEFAULT_IAM, split: ["train", "test", "val"] = 'train', partition: ["aachen", "original"] = 'aachen', mode: ["words", "lines"] = "words", image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        
        if split == 'train': 
            spl = 'tr.lst'

        elif split == 'test':
            spl = 'te.lst'

        else:
            spl = 'va.lst'

        parition_file = os.path.join(
            base_folder,
            'GT/partitions',
            partition, 
            spl
        )
        self.split = f"{split}_{partition}_{mode}"

        valid_records = [x.strip() for x in open(parition_file, 'r')]
        gt = [x.strip() for x in open(
            os.path.join(base_folder, 'GT', mode + '.txt')
        ) if x[0] != '#' and x.split()[1] == 'ok']

        self.data = []
        for sample in gt:
            
            sample = sample.split()
            file_id, ok,thr, x, y, w, h, _, transcription = sample[:8] + [' '.join(sample[8:])]
            author = file_id[:11] # TODO: Check this out dude
            if author in valid_records:
                folder, subfolder = file_id.split('-')[0], '-'.join(file_id.split('-')[:2])
                full_path = os.path.join(
                    base_folder,
                    mode,
                    folder,
                    subfolder,
                    file_id + '.png'
                )
                self.data.append(
                    {
                        "image_path": full_path,
                        "transcription": transcription if mode == 'words' else transcription.replace('|', ' ')
                    }
                )
        self.transforms = transforms
        self.patch_width = patch_width
        self.image_height = image_height
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        metadata = self.data[idx]
        
        
        image = Image.open(
                           
                           os.path.join(metadata['image_path'])
                           
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

    