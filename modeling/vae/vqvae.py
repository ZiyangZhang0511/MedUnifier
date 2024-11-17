import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoder
from . import quantizer
from . import decoder



class VQVAE(nn.Module):
    def __init__(self, vis_dim, txt_dim, feat_dim, codebook_size=512):
        """
        -vis_dim: feature dimensions of visual embeddings
        -txt_dim: feature dimensions of text embeddings
        -feat_dim: feature dimensions of latent vectors
        """
        super(VQVAE, self).__init__()

        self.mode = mode

        self.quantizer_top = quantizer.VectorQuantizer(codebook_size, feat_dim, mode=self.mode)
        self.quantizer_bottom = quantizer.VectorQuantizer(codebook_size, feat_dim*2, mode=self.mode)

        self.top_bridge = encoder.TopBridge2D(feat_dim)
        self.bottom_bridge = encoder.BottomBridge2D(vis_dim, txt_dim, feat_dim)

        self.decoder_top = decoder.VQDecoder2D(feat_dim, feat_dim*2, feat_dim, num_reslayers=1, level="top -> bottom")
        self.decoder_bottom = decoder.VQDecoder2D(feat_dim*3, feat_dim*3, 3, num_reslayers=1, level="bottom -> image")

        self.upsample_zq_t2b = nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=4, stride=2, padding=1)

        self.downsample_zq = nn.Conv2d(feat_dim*3, feat_dim*3, kernel_size=3, stride=1, padding=0)


    def forward(self, local_visual_embeddings, text_embeddings, query_tokens):

        z_e_bottom = self.bottom_bridge(local_visual_embeddings, text_embeddings)
        z_e_top = self.top_bridge(query_tokens)

        z_q_top, loss_vq_top, perplexity_top, id_top = self.quantizer_top(z_e_top)
        z_q_top2bottom = self.decoder_top(z_q_top)
        z_e_bottom_fused = torch.cat([z_e_bottom, z_q_top2bottom], dim=1)
        z_q_bottom, loss_vq_bottom, perplexity_bottom, id_bottom = self.quantizer_bottom(z_e_bottom_fused)

        z_q_top_upsampled = self.upsample_zq_t2b(z_q_top)

        z_q = torch.cat([z_q_top_upsampled, z_q_bottom], dim=1)

        z_q = self.downsample_zq(z_q)

        loss_codebook = loss_vq_bottom + loss_vq_top

        x_recon = self.decoder_bottom(z_q)
    

        return x_recon, loss_codebook, perplexity_top, perplexity_bottom
    
    @torch.no_grad()
    def get_ids(self, local_visual_embeddings, text_embeddings, query_tokens):
        z_e_bottom = self.bottom_bridge(local_visual_embeddings, text_embeddings)
        z_e_top = self.top_bridge(query_tokens)

        z_q_top, loss_vq_top, perplexity_top, id_top = self.quantizer_top(z_e_top)
        z_q_top2bottom = self.decoder_top(z_q_top)
        z_e_bottom_fused = torch.cat([z_e_bottom, z_q_top2bottom], dim=1)
        z_q_bottom, loss_vq_bottom, perplexity_bottom, id_bottom = self.quantizer_bottom(z_e_bottom_fused)
    
        return id_top, id_bottom

    def decode_code(self, code_top, code_bottom):
        z_q_top = self.quantizer_top.get_quantized_vectors(code_top)
        z_q_bottom = self.quantizer_bottom.get_quantized_vectors(code_bottom)

        z_q_top_upsampled = self.upsample_zq_t2b(z_q_top)
        z_q = torch.cat([z_q_top_upsampled, z_q_bottom], dim=1)
        z_q = self.downsample_zq(z_q)

        x_recon = self.decoder_bottom(z_q)

        return x_recon.detach().cpu()


