import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):

    def __init__(self, num_e, e_dim, beta=0.5):

        super(VectorQuantizer, self).__init__()

        self.num_e = num_e
        self.e_dim = e_dim
        self.beta = beta
        self.mode = mode

        self.codebook = nn.Embedding(self.num_e, self.e_dim)
        self.codebook.weight.data.uniform_(-1/num_e, 1/num_e)

    def forward(self, z_e):

        z_e = z_e.permute(0, 2, 3, 1).contiguous()

        # flatten z
        z_flatten = z_e.view(-1, self.e_dim)

        # distances from z to e_j: (z - e)^2 = z^2 + e^2 - 2*z*e
        distances = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook.weight**2, dim=1) - \
                    2 * torch.matmul(z_flatten, self.codebook.weight.t())
        
        # find the closest e within codebook
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print(min_encoding_indices.size())
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_e
        ).to(z_e.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.codebook.weight).view(z_e.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z_e) ** 2) + self.beta * torch.mean((z_q - z_e.detach()) ** 2)

        # preserve gradients
        z_q = z_e + (z_q - z_e).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match z shape

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, perplexity.detach().cpu(), min_encoding_indices.view(*z_e.shape[:-1])

    def get_quantized_vectors(self, min_encoding_indices):
        shape = min_encoding_indices.shape
        # min_encoding_indices (B, H, W) -> (B*H*W, 1)
        min_encoding_indices = min_encoding_indices.view(-1, 1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_e
        ).to(min_encoding_indices.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.codebook.weight).view(*shape, -1) # (B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        
        return z_q
        

