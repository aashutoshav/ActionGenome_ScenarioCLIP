from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class ActionGenomeDataset(Dataset):
    def __init__(self, metadata, transform=T.PILToTensor()):
        self.preprocess = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self.metadata[idx]
        img = Image.open(data['image_path']).convert("RGB")
        image = self.preprocess(img)
        caption = data['action']
        object_names = [x[0] for x in data['objects']]
        bboxes = [x[1] for x in data['objects']]
        objects_cropped = []
        for bbox in bboxes:
            bbox = [int(b) for b in bbox]
            img1 = Image.new(img.mode, img.size, color='black')
            img1.paste(img.crop(bbox), bbox)
            img1 = self.preprocess(img1)
            objects_cropped.append(img1)
        relation_images_list = [x[0] for x in data['relations']]
        relation_captions = [x[1] for x in data['relations']]
        relation_images = []
        for focused_region in relation_images_list:
            img1 = Image.open(focused_region)
            img1 = self.preprocess(img1)
            relation_images.append(img1)
        return image, caption, object_names, objects_cropped, relation_captions, relation_images
