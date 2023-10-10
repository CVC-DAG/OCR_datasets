from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_HIERTEXT = "/data/users/amolina/OCR/HierText"

def bbx_from_vertices_list(vertices):
    
    # Care about the index, potential source of errors
    return min(vertices, key = lambda x: x[0])[0], min(vertices,  key = lambda x: x[1])[1], max(vertices,  key = lambda x: x[0])[0], max(vertices,  key = lambda x: x[1])[1]

class HierTextDataset(GenericDataset):
    name = 'hiertext_dataset'
    def __init__(self, base_folder = DEFAULT_HIERTEXT, split: ['train', 'val'] = 'train',
                 handwritten = [True, False], legibility = [True, False], mode = 'words',
                 image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        
        self.image_height = image_height
        self.patch_width = patch_width
        self.transforms = transforms
        self.split = f"{split}_legibility-{legibility}_handwritten-{handwritten}"
        
        annotation_file = json.load(
            
            open(
                os.path.join(base_folder, 'gt', 'train.jsonl' if split == 'train' else 'validation.jsonl'), 'r'
            )
               
        )
        images_path = os.path.join(
            base_folder, 'train' if split == 'train' else 'validation'
        )
        self.base_images = images_path

        
        self.samples = []
        for num, annotation in enumerate(annotation_file['annotations']):
            image_path = os.path.join(images_path, annotation['image_id'] + '.jpg')
            for paragraph in annotation['paragraphs']:
                for line in paragraph['lines']:
                    x, y, x2, y2 = bbx_from_vertices_list(line['vertices'])
                    
                    if x2-x < y2 - y: continue # TODO: Evaluation without vertical lines. Not fair.
                    if line['legible'] in legibility and line['handwritten'] in handwritten and not line['vertical']:
                        
                        if mode == 'lines':
                            
                            self.samples.append({
                                
                                'bbx': bbx_from_vertices_list(line['vertices']),
                                'image_path': image_path,
                                'transcription': line['text'],
                                'vertical': line['vertical']

                                })
                        else:
                                for word in line['words']:
                                    if not word['vertical']:

                                        self.samples.append({
                                            
                                            'bbx': bbx_from_vertices_list(word['vertices']),
                                            'image_path': image_path,
                                            'transcription': word['text'],
                                            'vertical': word['vertical']
                                            
                                            })
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        metadata = self.samples[idx]

        image = Image.open(
                           
                           os.path.join(self.base_images, metadata['image_path'])
                           
                           ).crop(metadata['bbx']).convert('RGB')
        
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
            'split': self.split,
            'vertical': metadata['vertical']
        }
            