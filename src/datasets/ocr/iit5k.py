from PIL import Image
import os
import json 
import scipy.io

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_IIIT = "/data/users/amolina/OCR/IIIT5K/"

class IIIT5kDataset(GenericDataset):
    name = 'iiit5k_dataset'
    def __init__(self, base_folder = DEFAULT_IIIT, split: ["train", "test"] = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()

        matlab_filename = f"{split}data.mat"
        file = os.path.join(base_folder, matlab_filename)
        anns = scipy.io.loadmat(
            file
        )

        self.data = []
        for word in anns[matlab_filename.split('.')[0]][0]:

            self.data.append(
                {
                    "image_path": os.path.join(base_folder, word['ImgName'][0]),
                    "transcription": word['GroundTruth'][0]
                }
                )
        
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms

        self.split = split

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