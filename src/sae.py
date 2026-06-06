import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

class SparseAutoencoder(nn.Module):
    """
    A Sparse Autoencoder (SAE) PyTorch Module designed to reconstruct activations
    and enforce sparsity in the hidden feature activations via L1 regularization.
        
    Citations:
        - Bricken et al. [2023] ("Towards Monosemanticity")
        - Cunningham et al. [2024] ("Sparse Autoencoders Find Highly Interpretable Features")
    """
    def __init__(self, d_model: int = 768, expansion_factor: int = 8, tied: bool = False):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.hidden_dim = d_model * expansion_factor
        self.tied = tied

        # decoder bias (pre-bias subtraction / post-reconstruction shift)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # encoder projection (maps activation to overcomplete feature space)
        self.encoder = nn.Linear(d_model, self.hidden_dim, bias=True)
        nn.init.zeros_(self.encoder.bias) # init encoder bias to 0
        
        # decoder projection (maps sparse features back to original activation space)
        if self.tied:
            self.decoder = None
        else:
            self.decoder = nn.Linear(self.hidden_dim, d_model, bias=False)
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="relu") # init decoder w kaiming
            
        self.normalize_decoder_weights()

    @property
    def W_dec(self) -> torch.Tensor:
        """Returns the decoder weight matrix of shape [d_model, hidden_dim]."""
        if self.tied:
            return self.encoder.weight.t()
        return self.decoder.weight

    @property
    def W_enc(self) -> torch.Tensor:
        """Returns the encoder weight matrix of shape [hidden_dim, d_model]."""
        return self.encoder.weight

    @property
    def b_enc(self) -> torch.Tensor:
        """Returns the encoder bias of shape [hidden_dim]."""
        return self.encoder.bias

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input activations to the overcomplete feature space and applies ReLU.
        
        Input:
            x [..., d_model]: Tensor of input activations.
        Output:
            f(x) [..., hidden_dim]: Tensor of sparse hidden feature activations.
        """
        # centering activation by subtracting decoder bias, as described in Bricken et al. [2023]
        x_centered = x - self.b_dec
        f = torch.relu(self.encoder(x_centered))
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original activations from sparse feature activations.
        
        Input:
            f [..., hidden_dim]: Sparse activations.
        Output:
            x_hat [..., d_model]: Reconstructed activations.
        """
        if self.tied:
            # for tied weights, perform matmul manually (align dimensions)
            x_reconstructed = torch.matmul(f, self.encoder.weight) + self.b_dec
        else:
            x_reconstructed = self.decoder(f) + self.b_dec
        return x_reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: encodes and then decodes input activations.
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """
        Normalizes the decoder columns to unit norm to prevent L1 shrinkage scaling.
        For Tied SAEs, we normalize the columns of self.encoder.weight.t() (which is encoder.weight).
        """
        if self.tied:
            # tied weights, normalize the rows of encoder.weight (which represent decoder columns)
            norms = torch.norm(self.encoder.weight, p=2, dim=1, keepdim=True)
            self.encoder.weight.div_(norms.clamp(min=1e-8))
        else:
            # normalize the columns of decoder.weight (dim=0 is the output dimension)
            # [d_model, hidden_dim], so dim=0 are cols.
            norms = torch.norm(self.decoder.weight, p=2, dim=0, keepdim=True)
            self.decoder.weight.div_(norms.clamp(min=1e-8))

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor, l1_coeff: float) -> Dict[str, torch.Tensor]:
        """
        Computes the reconstruction MSE and sparse L1 loss.
        
        Loss = MSE(x, x_hat) + l1_coeff * sum(|f|_1)
        """
        mse_loss = torch.mean((x - x_hat) ** 2)
        # sparsity l1 loss (avg sum of active feat activ across tokens)
        l1_loss = torch.mean(torch.sum(torch.abs(f), dim=-1))
        
        total_loss = mse_loss + l1_coeff * l1_loss
        
        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss
        }
