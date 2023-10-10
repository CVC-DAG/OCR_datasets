from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_WASHINGTON = "/data/users/amolina/OCR/GW"

class GWDataset(GenericDataset):
    name = 'gw_dataset'
    def __init__(self, base_folder = DEFAULT_WASHINGTON, split: ['train', 'test', 'val'] = 'train', cross_val = 'cv1', mode = 'word', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        
        self.split = split
        self.patch_width = patch_width
        self.image_height = image_height
        self.transforms = transforms
        self.fold = cross_val
        
        valid_lines = [x.strip() for x in
                            open(
                                os.path.join(base_folder, 'sets', cross_val, split + '.txt'), 'r'
                            ).readlines()
                        ]
        self.images_path = os.path.join(
            base_folder, 'data', f'{mode}_images_normalized'
        )
        
        line_groundtruth = [x.strip() for x in
                            open(
                                os.path.join(base_folder, 'ground_truth', 'transcription.txt'), 'r'
                            ).readlines()
                        ]
        samples = []
        for transcription in line_groundtruth:
            
            ident, line_transcription = transcription.split()
            
            
            line_transcription = line_transcription\
                .replace("s_pt", ".").replace("s_cm", ",")\
                .replace("s_mi", "-").replace("s_qo", ":").replace("s_sq", ";")\
                .replace("s_et", "V").replace("s_bl", "(").replace("s_br", ")")\
                .replace("s_qt", "'").replace("s_GW", "G.W.").replace("s_", "")
            
            if ident in valid_lines:
                if mode == 'word':
                    for counter, word in enumerate(line_transcription.split('|')):

                        impath = os.path.join(self.images_path, f"{ident}-{counter + 1:02d}.png")
                        if not os.path.exists(impath): raise FileNotFoundError(impath)
                        samples.append(
                            {
                                'image_path': impath,
                                'transcription': word.replace('-', '')
                            }
                        )
                
                else:
                    impath = os.path.join(self.images_path, f"{ident}.png")
                    if not os.path.exists(impath): raise FileNotFoundError(impath)
                    samples.append(
                        {
                            'image_path': impath,
                            'transcription': line_transcription.replace('|', ' ').replace('-', '')
                        }
                    )
        self.data = samples

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        metadata = self.data[idx]

        image = Image.open(metadata['image_path']).convert('RGB')
        
        original_width, _ = image.size
        new_width = original_width + (original_width % self.patch_width)
        
        image_resized = image.resize((new_width, self.image_height))
         
        input_tensor = self.transforms(image_resized)
        
        annotation = metadata['transcription']
        
        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": annotation,
            'dataset': self.name,
            'split': f"{self.fold}_{self.split}",
            'path': metadata['image_path']
        }
                        
                
                
        