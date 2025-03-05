import os
import PIL

import torch
import torchvision.transforms as transforms

from laplace.baselaplace import DiagLaplace


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


class DiffusionLLDiagLaplace(DiagLaplace):

    def __init__(
        self,
        model,
        f_preprocess_la_input,
        last_layer_name,
        backend,
        likelihood="regression",
        sigma_noise=1.0,
        prior_precision=1.0,
        prior_mean=0.0,
        temperature=1.0,
    ):
        sum_param = 0
        sum_param_grad = 0
        sum_param_final_layer = 0
        for name, param in model.named_parameters():
            sum_param += param.numel()
            if param.requires_grad:
                sum_param_grad += param.numel()
            if last_layer_name in name:
                sum_param_final_layer += param.numel()
            if not last_layer_name in name:
                param.requires_grad = False
        print(f"Total parameters: {sum_param}")
        print(f"Total parameters with grad: {sum_param_grad}")
        print(f"Total parameters in final layer: {sum_param_final_layer}")

        super().__init__(model, likelihood, sigma_noise, prior_precision, prior_mean, temperature, backend=backend)

        self.f_preprocess_la_input = f_preprocess_la_input

    def fit(
        self,
    ):
        raise NotImplementedError()

    def _curv_closure(
        self,
    ):
        raise NotImplementedError()
