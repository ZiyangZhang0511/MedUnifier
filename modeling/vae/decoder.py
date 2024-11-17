import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import ResidualStack

class VQDecoder2D(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_reslayers=1, level="top -> bottom"):
        super(VQDecoder2D, self).__init__()

        self.resdiual_stack = ResidualStack(in_dim, in_dim//2, num_reslayers=num_reslayers, mode="2D")

        if level == "top -> bottom":
            self.decoder = nn.Sequential(
                self.make_deconv_block(in_dim, out_dim, kernel_size=4, stride=2, padding=1, final_layer=True),
            )
        elif level == "bottom -> image":
             self.decoder = nn.Sequential(
                self.make_deconv_block(in_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
                self.make_deconv_block(hidden_dim*2, hidden_dim*2, kernel_size=4, stride=2, padding=1),
                self.make_deconv_block(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),
                self.make_deconv_block(hidden_dim, out_dim, kernel_size=4, stride=2, padding=1, final_layer=True),
            )

    def forward(self, x):
        x = self.resdiual_stack(x)
        x = self.decoder(x)
        return x

    def make_deconv_block(self, in_dim, out_dim, kernel_size, stride, padding, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
            )

class VQDecoder3D(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_reslayers=1, level="top -> bottom"):
        super(VQDecoder3D, self).__init__()

        self.resdiual_stack = ResidualStack(in_dim, in_dim//2, num_reslayers=num_reslayers, mode="3D")

        if level == "top -> bottom":
            self.decoder = nn.Sequential(
                self.make_deconv_block(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                self.make_deconv_block(in_dim, out_dim, kernel_size=4, stride=2, padding=1, final_layer=True),
            )
        elif level == "bottom -> image":
             self.decoder = nn.Sequential(
                self.make_deconv_block(in_dim, hidden_dim*2, kernel_size=(3, 4, 4), stride=(1, 2, 2) , padding=1),
                self.make_deconv_block(hidden_dim*2, hidden_dim*2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
                self.make_deconv_block(hidden_dim*2, hidden_dim, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
                self.make_deconv_block(hidden_dim, out_dim, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, final_layer=True),
            )
    
    def forward(self, x):
        x = self.resdiual_stack(x)
        x = self.decoder(x)
        return x

    def make_deconv_block(self, in_dim, out_dim, kernel_size, stride, padding, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride, padding),
                nn.BatchNorm3d(out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride, padding),
            )




