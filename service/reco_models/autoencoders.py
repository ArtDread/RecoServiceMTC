from __future__ import annotations

import dill
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn


class OnlineAE:
    """This class is implementation of recommendations generation with autoencoder.

    Attributes:
        device: The execution device, CPU only.
        enc_dims: The list containing encoder dimensions structure.
        model: The instance of the VariationalAE class.
        hot_users: The dictionary containing list of relevant items for each hot user.
        hot_users_weights: The dictionary with corresponding weight for each
            interaction.
        item_mapping: The dictionary to make the transition from internal
            (generated during the model fitting) to external user ids.
        item_inv_mapping: The dictionary to make the transition from external
            to internal user ids.
        filter_viewed: The flag to get rid of watched items in recos or not.

    """

    def __init__(
        self,
        ae_paths: tuple[str, str, str, str, str],
    ):
        (
            model_path,
            hot_users_path,
            hot_users_weights_path,
            item_mapping_path,
            enc_dims_path,
        ) = ae_paths
        self.device = torch.device("cpu")
        with open(enc_dims_path, "rb") as fp:
            self.enc_dims: list[int] = dill.load(fp)
        self._initialize_model(model_path)
        with open(hot_users_path, "rb") as f:
            self.hot_users: dict[int, list[int]] = dill.load(f)
        with open(hot_users_weights_path, "rb") as f:
            self.hot_users_weights: dict[int, list[float]] = dill.load(f)
        with open(item_mapping_path, "rb") as f:
            self.item_mapping: dict[int, int] = dill.load(f)
        self.item_inv_mapping = {v: k for k, v in self.item_mapping.items()}
        self.filter_viewed = True

    def _initialize_model(self, model_path) -> None:
        """Initialize ae model and get it prepared for inference."""
        self.model = VariationalAE(self.enc_dims)
        try:
            with open(model_path, "rb") as fp:
                state_dict = torch.load(fp, map_location=self.device)
        except FileNotFoundError:
            print("Run `make load_models` to load the pickled weights")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _create_user_input(self, input_dim, user_id) -> torch.Tensor:
        """Build proper user input for the autoencoder model.

        Extract interactions from data for the specific user id (the existence of
        the data is meant) and build input for the autoencoder model: tensor of
        shape (1, input_dim) filled with floats where 0.0 means absence of interaction.

        Args:
            input_dim: The input dimension of the model.
            user_id: The external user id.

        Returns:
            user_input: The user input for autoencoder model.

        """
        items = self.hot_users[user_id]
        ratings = self.hot_users_weights[user_id]

        user_input = torch.zeros(size=(1, input_dim), dtype=torch.float32)
        items = [self.item_mapping[item] for item in items]

        # Complete user input
        ratings_tsr = torch.tensor(ratings, dtype=torch.float32)
        user_input[:, items] = ratings_tsr
        return user_input

    def _predict_for_user_id(self, user_input, k_recs) -> NDArray[np.int64]:
        """Get top-k recommendations for one user.

        Args:
            user_input: The user input for autoencoder model.
            model: The allowed PyTorch model in evaluation regime.
            k: The number of recos which are considered.
            ind_to_item: The mapping from internal item ids to the external.
            filter_viewed: The flag to get rid of watched items in recos or not.

        Returns:
            recos_at_k: The top-k recommendations for user as external item indices.

        """
        # Get recos_at_k
        recon: NDArray[np.float32] = (
            self.model.predict(user_input).detach().to("cpu").numpy()
        )
        if self.filter_viewed:
            recon[user_input.numpy().nonzero()] = float("-inf")
        recos: NDArray[np.int64] = np.argpartition(-recon, k_recs, axis=1)
        recos_top_k = recos[:, :k_recs]
        # Internal indices to external
        recos_top_k = np.vectorize(self.item_inv_mapping.get)(recos_top_k)
        return recos_top_k.squeeze()

    def predict(self, user_id: int, k_recs: int) -> None | list[int]:
        """Return top k_recs recos for specific user_id using ae model."""
        # Get recommendations
        if user_id not in self.hot_users:
            return None
        input_dim = self.enc_dims[0]
        user_input = self._create_user_input(input_dim, user_id)
        recos = self._predict_for_user_id(user_input, k_recs)
        return list(recos)


class VariationalAE(nn.Module):
    """The realization of a variational autoencoder using PyTorch.

    Variational autoencoder based on multinomial likelihood with noising
    potentially making it robust.

    Attributes:
        enc_dims: The list containing dimension sizes of encoder layers.
        dec_dims: The list containing dimension sizes of decoder layers.
        corrupt_ratio: The denoising parameter, i.e. the proportion of
            data that will be corrupted.

    """

    def __init__(
        self,
        enc_dims: list[int],
        dec_dims: list[int] | None = None,
        corrupt_ratio: float = 0.2,
    ):
        super().__init__()
        if not 0 < corrupt_ratio < 1:
            raise ValueError(
                f"corrupt_ratio={corrupt_ratio} should be a float in (0, 1) range."
            )

        self.enc_dims = enc_dims
        if dec_dims:
            if enc_dims[0] != dec_dims[-1]:
                raise ValueError("Reconstruction dimension should be equal to input")
            if enc_dims[-1] != dec_dims[0]:
                raise ValueError(
                    "Latent dimension for encoder and decoder should be equal"
                )
            self.dec_dims = dec_dims
        else:
            # Decoder is symmetric by default
            self.dec_dims = enc_dims[::-1]

        # Create dimension to store the params of variational distribution
        enc_dims_ext = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        self.enc_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(enc_dims_ext[:-1], enc_dims_ext[1:])
            ]
        )
        self.dec_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])
            ]
        )

        self.corrupt_ratio = corrupt_ratio
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.enc_layers + self.dec_layers:
            # Xavier Initialization
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            scope = np.sqrt(6.0 / (fan_in + fan_out))
            layer.weight.data.uniform_(-scope, scope)
            layer.bias.data.zero_()

    @property
    def device(self):
        """Move all model parameters to the device."""
        return next(self.parameters()).device

    def forward(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build data reconstruction using latent representation."""
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Get prediction while inference."""
        mu, _ = self.encode(batch)
        return self.decode(mu)

    def encode(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain params of the variational distribution."""
        # Row-wise L2 norm is meant to take user's rating behaviour into account
        output = F.normalize(batch)
        # Corrupt the interactions matrix only during training
        output = F.dropout(batch, p=self.corrupt_ratio, training=self.training)

        for i, layer in enumerate(self.enc_layers):
            output = layer(output)
            # Apply activation except the last layer
            if i != len(self.enc_layers) - 1:
                output = torch.tanh(output)
            else:
                mu = output[:, : self.enc_dims[-1]]
                logvar = output[:, self.enc_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Obtain a latent vector by applying a reparametrization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Build reconstruction from the latent representation."""
        output = z
        for i, layer in enumerate(self.dec_layers):
            output = layer(output)
            if i != len(self.dec_layers) - 1:
                output = torch.tanh(output)
        return output
