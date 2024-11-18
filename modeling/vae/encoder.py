import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import (
    PositionalEncoding1D, 
    PositionalEncodingPermute2D, 
    PositionalEncodingPermute3D, 
    Summer,
    )

from .residual import ResidualStack


class TopBridge2D(nn.Module):

    def __init__(self, feat_dim):
        super(TopBridge2D, self).__init__()

        self.feat_dim = feat_dim

        self.pe2D_summer = Summer(PositionalEncodingPermute2D(feat_dim))

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim*2, feat_dim),
        )

        self.resdiual_stack = ResidualStack(feat_dim, feat_dim//2, num_reslayers=3, mode="2D")

    def forward(self, query_tokens):

        query_tokens = self.mlp(query_tokens)

        query_tokens = query_tokens.permute(0, 2, 1).contiguous()
        query_tokens = query_tokens.view(-1, self.feat_dim, 4, 8)

        ### add pe2D for query tokens ###
        query_tokens = self.pe2D_summer(query_tokens)

        ### upsample query tokens to [B, 768, 8, 8] ###
        query_tokens_upsampled = F.interpolate(query_tokens, [8, 8])

        ### go through residual stack ###
        z_e_top = self.resdiual_stack(query_tokens_upsampled)

        return z_e_top


class BottomBridge2D(nn.Module):
    def __init__(self, vis_dim, txt_dim, feat_dim):
        super(BottomBridge2D, self).__init__()

        self.feat_dim = feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(vis_dim+txt_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        self.pe2D_summer = Summer(PositionalEncodingPermute2D(feat_dim))

        self.resdiual_stack = ResidualStack(feat_dim, feat_dim//2, num_reslayers=3, mode="2D")

    def forward(self, local_visual_embeddings, text_embedding):

        ### concatenate local_visual_embeddings and text_embeddings ###
        local_visual_embeddings = local_visual_embeddings.view(-1, 16, 16, local_visual_embeddings.shape[-1])
        batch_size, h, w, visual_dim = local_visual_embeddings.size()
        text_embedding = text_embedding.unsqueeze(dim=1).expand(-1, h, w, -1)
        concat_embeddings = torch.cat([local_visual_embeddings, text_embedding], dim=-1)

        ### fuse vision and text info by MLP ###
        fused_embeddings = self.mlp(concat_embeddings) # (B, h, w, feat_dim)
    
        ### add positional encoding ###
        fused_embeddings = fused_embeddings.permute(0, 3, 1, 2).contiguous()
        fused_embeddings = self.pe2D_summer(fused_embeddings) # (B, feat_dim, h, w)

        ### go through residual stack ###
        z_e_bottom = self.resdiual_stack(fused_embeddings)

        return z_e_bottom


class TopBridge3D(nn.Module):

    def __init__(self, feat_dim):
        super(TopBridge3D, self).__init__()

        self.feat_dim = feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim*2, feat_dim),
        )

        self.pe3D_summer = Summer(PositionalEncodingPermute3D(feat_dim))

        self.resdiual_stack = ResidualStack(feat_dim, feat_dim//2, num_reslayers=1, mode="3D")

    def forward(self, query_tokens):

        query_tokens = self.mlp(query_tokens)
        query_tokens = query_tokens.permute(0, 2, 1).contiguous()

        query_tokens = query_tokens.view(-1, self.feat_dim, 2, 4, 4)

        ### add pe3D for query tokens ###
        query_tokens = self.pe3D_summer(query_tokens)

        ### go through residual stack ###
        z_e_top = self.resdiual_stack(query_tokens)

        return z_e_top

class BottomBridge3D(nn.Module):

    def __init__(self, vis_dim, txt_dim, feat_dim):
        super(BottomBridge3D, self).__init__()

        self.feat_dim = feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(vis_dim+txt_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        self.pe3D_summer = Summer(PositionalEncodingPermute3D(feat_dim))

        self.resdiual_stack = ResidualStack(feat_dim, feat_dim//2, num_reslayers=1, mode="3D")

    def forward(self, local_visual_embeddings, text_embedding):

        ### concatenate local_visual_embeddings and text_embeddings ###
        local_visual_embeddings = local_visual_embeddings.view(-1, 8, 16, 16, local_visual_embeddings.shape[-1])
        batch_size, d, h, w, visual_dim = local_visual_embeddings.size()
        text_embedding = text_embedding.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, d, h, w, -1)
        # print(text_embeddings.size())
        # print(local_visual_embeddings.size())
        concat_embeddings = torch.cat([local_visual_embeddings, text_embedding], dim=-1)

        ### fuse vision and text info by MLP ###
        fused_embeddings = self.mlp(concat_embeddings) # (B, d, h, w, feat_dim)
    
        ### add positional encoding ###
        fused_embeddings = fused_embeddings.permute(0, 4, 1, 2, 3).contiguous()
        fused_embeddings = self.pe3D_summer(fused_embeddings) # (B, feat_dim, h, w)

        ### go through residual stack ###
        z_e_bottom = self.resdiual_stack(fused_embeddings)

        return z_e_bottom
