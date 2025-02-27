import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop
import PIL

from typing import Any
from collections.abc import MutableMapping
from torch.nn.utils import parameters_to_vector
from laplace.baselaplace import DiagLaplace
from laplace.curvature.curvlinops import CurvlinopsEF

import torch.nn.functional as F


def list_image_files_and_class_recursively(image_path):
    paths = []
    classes= []
    classes_name = []
    i=0
    for subentry in os.listdir(image_path):
        subfull_path = os.path.join(image_path, subentry)
        for entry in os.listdir(subfull_path):
            full_path = os.path.join(subfull_path, entry)
            paths.append(full_path)
            classes.append(i)
            classes_name.append(subentry)
        i +=1

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
        return len(self.paths)//self.train_la_data_size

    def __getitem__(self, idx):
        
        rand_id = idx * torch.randint(low=1, high=self.train_la_data_size, size=(1,))[0]
        x_path = self.paths[rand_id]
        label = self.classes[rand_id]

        x = PIL.Image.open(x_path).convert('RGB')
        x = self.transform(x)

        return  x, label


class DiffusionCurvlinopsEF(CurvlinopsEF):

    def gradients(
        self, x, y, t, labels,
    ):
        """Compute batch gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at
        current parameter \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        loss : torch.Tensor
        """
        
        def loss_single(x, y, t, labels, params_dict, buffers_dict):
            """Compute the gradient for a single sample."""
            x, y = x.unsqueeze(0), y.unsqueeze(0)  # vmap removes the batch dimension
            t, labels = t.unsqueeze(0), labels.unsqueeze(0)

            output = torch.func.functional_call(
                self.model, 
                (params_dict, buffers_dict), 
                #(t, x, labels)
                (x, t, labels)
            )
            output = torch.split(output, 3, dim=1)[0]  # get rid of CFG channels

            loss = torch.func.functional_call(self.lossfunc, {}, (output, y))
            return loss, loss

        grad_fn = torch.func.grad(loss_single, argnums=4, has_aux=True)
        batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, 0, None, None))

        batch_grad, batch_loss = batch_grad_fn(
            x, y, t, labels, self.params_dict, self.buffers_dict
        )
        Gs = torch.cat([bg.flatten(start_dim=1) for bg in batch_grad.values()], dim=1)

        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]

        loss = batch_loss.sum(0)

        return Gs, loss

    def diag(
        self,
        x,
        y,
        t,
        labels,
        **kwargs,
    ):
        # Gs is (batchsize, n_params)
        Gs, loss = self.gradients(x, y, t, labels)
        Gs, loss = Gs.detach(), loss.detach()
        diag_ef = torch.einsum("bp,bp->p", Gs, Gs)
        return self.factor * loss, self.factor * diag_ef

    
class DiffusionLLDiagLaplace(DiagLaplace):

    def __init__(self, model, f_preprocess_la_input, last_layer_name,likelihood='regression', sigma_noise=1., prior_precision=1., prior_mean=0.0, 
                      temperature=1.0, backend=DiffusionCurvlinopsEF,):
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


    def fit(self, train_loader, override=True):
        if override:
            self._init_H()
            self.loss = 0
            self.n_data = 0

        self.model.eval()
        self.mean: torch.Tensor = parameters_to_vector(self.params)
        if not self.enable_backprop:
            self.mean = self.mean.detach()

        # print(f"Mean: {self.mean.shape}")
        # print(f"H: {self.H.shape}")
        
        X, y = next(iter(train_loader))

        (t, X, labels), _ = self.f_preprocess_la_input(X, y, self._device)
        with torch.no_grad():
            out = self.model(X, t, labels)
        out = out.view(out.size(0), -1)  # flatten to (B, -1)
        self.n_outputs = out.shape[-1]
        setattr(self.model, 'output_size', self.n_outputs)

        N = len(train_loader.dataset)
        i=0
        for X, y in train_loader:
            print(i)
            (t, X, labels), y = self.f_preprocess_la_input(X, y, self._device)
            X, labels, t, y = X.to(self._device), labels.to(self._device), t.to(self._device), y.to(self._device)

            self.model.zero_grad()
            loss_batch, H_batch = self._curv_closure(X, y, t, labels, N)
            self.loss += loss_batch
            self.H += H_batch
            i+=1

        self.n_data += N

    
    def _curv_closure(
        self,
        X,
        y,
        t,
        labels,
        N,
    ):
        return self.backend.diag(X, y, t, labels, N=N, **self._asdl_fisher_kwargs)


def preprocess_la_adm(x, y, betas, num_timesteps, device, dtype=torch.float32):

    x = x.to(device, dtype=dtype, non_blocking=True)
    y = y.to(device, non_blocking=True)
    t = torch.randint(low=0, high=num_timesteps, size=(x.shape[0],), device=device)
    e = torch.randn_like(x, device=device, dtype=dtype)
    b = betas.to(device, dtype=dtype)
    a = (1-b).cumprod(dim=0)[t]
    xt = x * a[:, None, None, None].sqrt() + e * (1.0 - a[:, None, None, None]).sqrt()

    return  (t, xt, y), e