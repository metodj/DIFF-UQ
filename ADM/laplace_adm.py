import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from torch.nn.utils import parameters_to_vector
from laplace.curvature.curvlinops import CurvlinopsEF

from utils import DiffusionLLDiagLaplace


class DiffusionCurvlinopsEF(CurvlinopsEF):

    def gradients(
        self,
        x,
        y,
        t,
        labels,
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
                # (t, x, labels)
                (x, t, labels),
            )
            output = torch.split(output, 3, dim=1)[0]  # get rid of CFG channels

            loss = torch.func.functional_call(self.lossfunc, {}, (output, y))
            return loss, loss

        grad_fn = torch.func.grad(loss_single, argnums=4, has_aux=True)
        batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, 0, None, None))

        batch_grad, batch_loss = batch_grad_fn(x, y, t, labels, self.params_dict, self.buffers_dict)
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


class ADMLLDiagLaplace(DiffusionLLDiagLaplace):

    def __init__(
        self,
        model,
        f_preprocess_la_input,
        last_layer_name,
        backend=DiffusionCurvlinopsEF,
        likelihood="regression",
        sigma_noise=1.0,
        prior_precision=1.0,
        prior_mean=0.0,
        temperature=1.0,
    ):
        super().__init__(
            model,
            f_preprocess_la_input,
            last_layer_name,
            backend,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
        )

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
        setattr(self.model, "output_size", self.n_outputs)

        N = len(train_loader.dataset)
        i = 0
        for X, y in train_loader:
            print(i)
            (t, X, labels), y = self.f_preprocess_la_input(X, y, self._device)
            X, labels, t, y = X.to(self._device), labels.to(self._device), t.to(self._device), y.to(self._device)

            self.model.zero_grad()
            loss_batch, H_batch = self._curv_closure(X, y, t, labels, N)
            self.loss += loss_batch
            self.H += H_batch
            i += 1

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
    a = (1 - b).cumprod(dim=0)[t]
    xt = x * a[:, None, None, None].sqrt() + e * (1.0 - a[:, None, None, None]).sqrt()

    return (t, xt, y), e
