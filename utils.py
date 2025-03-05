import os
import PIL

import torch
import torchvision.transforms as transforms


def list_image_files_and_class_recursively(image_path):
    paths = []
    classes = []
    classes_name = []
    i = 0
    for subentry in os.listdir(image_path):
        subfull_path = os.path.join(image_path, subentry)
        for entry in os.listdir(subfull_path):
            full_path = os.path.join(subfull_path, entry)
            paths.append(full_path)
            classes.append(i)
            classes_name.append(subentry)
        i += 1

    return paths, classes, classes_name


class LaplaceDataset(torch.utils.data.Dataset):

    def __init__(self, device, image_path, image_size=256, train_la_data_size=50):

        super().__init__()
        self.device = device

        self.image_path = image_path
        self.paths, self.classes, _ = list_image_files_and_class_recursively(self.image_path)
        self.image_size = image_size
        self.train_la_data_size = train_la_data_size

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.paths) // self.train_la_data_size

    def __getitem__(self, idx):

        rand_id = idx * torch.randint(low=1, high=self.train_la_data_size, size=(1,))[0]
        x_path = self.paths[rand_id]
        label = self.classes[rand_id]

        x = PIL.Image.open(x_path).convert("RGB")
        x = self.transform(x)

        return x, label
