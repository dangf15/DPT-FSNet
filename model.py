
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from utils import show_params, show_model
from conv_stft import ConvSTFT, ConviSTFT

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        if bidirectional:
            self.linear2 = Linear(d_model*2*2, d_model)
        else:
            self.linear2 = Linear(d_model*2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Dual_Transformer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size//2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, t, f]
        b, c, t, f = input.shape
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(2, 0, 3, 1).contiguous().view(t, b*f, -1)  # [t, b*f, c]
            row_output = self.row_trans[i](row_input)  
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, t, f]
            row_output = self.row_norm[i](row_output)  
            output = output + row_output  

            col_input = output.permute(3, 0, 2, 1).contiguous().view(f, b*t, -1)  # [f, b*t, c]
            col_output = self.col_trans[i](col_input)  
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, t, f]
            col_output = self.col_norm[i](col_output)  
            output = output + col_output  

        del row_input, row_output, col_input, col_output
        output = self.output(output)  # [b, c, t, f]

        return output



class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out



class Net(nn.Module):
    def __init__(self, L=512, width=64):
        super(Net, self).__init__()
        self.L = L
        self.frame_shift = self.L // 2
        self.N = 256
        self.B = 256
        self.H = 512
        self.P = 3
        # self.device = device
        self.in_channels = 2
        self.out_channels = 2
        self.kernel_size = (2, 3)
        # self.elu = nn.SELU(inplace=True)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.width = width

        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 256]
        self.inp_norm = nn.LayerNorm(256)
        self.inp_prelu = nn.PReLU(self.width)

        self.enc_dense1 = DenseBlock(256, 4, self.width)

        self.dual_transformer = Dual_Transformer(64, 64, num_layers=4)  # # [b, 64, nframes, 8]

        # gated output layer
        self.output1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Tanh()
        )
        
        self.output2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Sigmoid()
        )

        self.dec_dense1 = DenseBlock(256, 4, self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))
        self.stft =  ConvSTFT(self.L, self.frame_shift, self.L, 'hanning', 'real', fix=True)
        self.istft = ConviSTFT(self.L, self.frame_shift, self.L, 'hanning', 'complex', fix=True)

        show_model(self)
        show_params(self)

    def forward(self, x, masking_mode='C'):


        x = self.stft(x)
        real = x[0]
        imag = x[1]
        x = torch.stack([real,imag],1)
        x = x.permute(0,1,3,2) # [B, 2, num_frames, num_bins]
        x = x[...,1:]
        #print(x.shape)

        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, num_frames, frame_size]
        x1 = self.enc_dense1(out)   # [b, 64, num_frames, frame_size]
        out = self.dual_transformer(x1)  # [b, 64, num_frames, 256]
        out = self.output1(out) * self.output2(out)  # mask [b, 64, num_frames, 256]
        out = self.dec_dense1(out)
        out = self.out_conv(out)

        real = x[:,0]
        imag = x[:,1]

        mask_real = out[:,0]
        mask_imag = out[:,1]
                                                                            

        if masking_mode == 'E' :
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2( imag_phase, real_phase )
            #mask_mags = torch.clamp_(mask_mags,0,100) 
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase)
        elif masking_mode == 'C':
            real,imag = real*mask_real-imag*mask_imag, real*mask_imag+imag*mask_real
        elif masking_mode == 'R':
            real, imag = real*mask_real, imag*mask_imag

        #print(out.shape)

        real = torch.cat((torch.zeros((real.size()[0], real.size()[1], 1)).to(device='cuda'), real), -1)
        imag = torch.cat((torch.zeros((imag.size()[0], imag.size()[1], 1)).to(device='cuda'), imag), -1)
        out = torch.cat([real,imag],-1).permute(0,2,1)

        out = self.istft(out)

        return out                                                                       
