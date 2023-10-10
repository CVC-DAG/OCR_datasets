from PIL import Image
import os

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_ESPOSALLES = "/data/users/amolina/OCR/ESPOSALLES"

class EsposalledDataset(GenericDataset):
    name = 'esposalles_dataset'

    def __init__(self, base_folder = DEFAULT_ESPOSALLES, split = 'train', cross_val = 'cv1', mode = 'words', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        #### ESPOSALLES DATASET ####
        # base_folder: folder with train - test splits.
        # split: train or test
        # cross-val: cross validation split
        # mode: words or lines.
        # image_height: height of the image sequence. Resize to this height.
        # patch_width: width of the sliding window. Resize sequence to closest superior multiple.
        ############################

        self.image_height = image_height
        self.patch_width = patch_width

        self.base_folder = os.path.join(base_folder, split)
        valid_records = [record.strip() for record in open(os.path.join(base_folder, 'splits', cross_val, split + '.txt'), 'r').readlines()]

        records = [{'folder': os.path.join(self.base_folder, page_id, mode),
                    'transcription_file': os.path.join(self.base_folder, page_id, mode, page_id + '_transcription.txt'),
                    'page_id': page_id} for page_id in os.listdir(self.base_folder) if page_id in valid_records and 'idPage' in page_id] # TODO: Check there's no leak and it's getting properly filtered
        samples = {}
        for record_folder in records:
            
            files = os.listdir(record_folder['folder'])
            transcriptions = {os.path.join(record_folder['folder'], 
                                           f'{line.strip().split(":")[0]}.png'): line.strip().split(":")[1] 
                                            for line in open(record_folder['transcription_file'], 'r').readlines()
                                            if f'{line.strip().split(":")[0]}.png' in files}
            samples = {**samples, **transcriptions}
        
        self.samples = samples
        self.keys = list(self.samples.keys())
        self.transforms = transforms
        self.fold = cross_val
        self.split = split
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        file_path = self.keys[idx]
        image = Image.open(file_path).convert('RGB')
        
        original_width, _ = image.size
        new_width = original_width + (original_width % self.patch_width)
        
        image_resized = image.resize((new_width, self.image_height))
         
        input_tensor = self.transforms(image_resized)
        
        annotation = self.samples[file_path]
        
        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": annotation,
            'dataset': self.name,
            'split': f"{self.fold}_{self.split}"
        }