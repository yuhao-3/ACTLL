import torch
import torch.nn.functional as F
import math 
from src.models.MultiTaskClassification import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import numpy as np

#########################################################################################################
# META CLASS
#########################################################################################################
class MetaAE(nn.Module):
    def __init__(self, name='AE'):
        super(MetaAE, self).__init__()

        self.encoder = None
        self.decoder = None

        self.name = name

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, x):
        # Not Used for now
        emb = self.encoder(x)
        emb = emb / torch.sqrt(torch.sum(emb ** 2, 2, keepdim=True))
        return emb

    def get_name(self):
        return self.name


#########################################################################################################
# CONVOLUTIONAL AUTOENCODER
#########################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, dropout=0.2, normalization='none'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.conv, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        self.convtraspose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                               output_padding=output_padding,
                                               padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.convtraspose, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        num_blocks = len(num_channels)
        layers = []
        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dropout=dropout,
                          normalization=normalization)]

        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(num_channels[-1], embedding_dim, 1)

    def forward(self, x):
        # print("x0 shape = ", x.shape)
        x = self.network(x.transpose(2, 1))
        # print("x1 shape = ", x.shape, x.data.shape[2], x.data.shape)
        x = F.max_pool1d(x, kernel_size=x.data.shape[2])
        # print("x2 (max_pool1d) shape = ", x.shape)
        x = self.conv1x1(x)
        # print("x3 (emd) shape = ", x.shape)
        return x


def conv_out_len(seq_len, ker_size, stride, padding, dilation, stack):
    for _ in range(stack):
        seq_len = int((seq_len + 2 * padding - dilation * (ker_size - 1) - 1) / stride + 1)
    return seq_len


class ConvDecoder(nn.Module):
    def __init__(self, embedding_dim, num_channels, seq_len, out_dimension, kernel_size, stride=2, padding=0,
                 dropout=0.2, normalization='none'):
        super().__init__()

        num_channels = num_channels[::-1]
        num_blocks = len(num_channels)

        self.compressed_len = conv_out_len(seq_len, kernel_size, stride, padding, 1, num_blocks)

        # Pad sequence to match encoder lenght
        if stride > 1:
            output_padding = []
            seq = seq_len
            for _ in range(num_blocks):
                output_padding.append(seq % 2)
                seq = conv_out_len(seq, kernel_size, stride, padding, 1, 1)
            # bit flip
            if kernel_size % 2 == 1:
                output_padding = [1 - x for x in output_padding[::-1]]
            else:
                output_padding = output_padding[::-1]
        else:
            output_padding = [0] * num_blocks

        layers = []
        for i in range(num_blocks):
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvTransposeBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding[i], dropout=dropout, normalization=normalization)]
        self.network = nn.Sequential(*layers)
        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(num_channels[-1], out_dimension, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.network(x)
        x = self.conv1x1(x)
        return x.transpose(2, 1)


class CNNAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_size, dropout,
                 normalization=None, stride=2, padding=0, name='CNN_AE'):
        super(CNNAE, self).__init__(name=name)

        self.encoder = ConvEncoder(input_size, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)
        self.decoder = ConvDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)
        
        
        


######################### DIFFUSION MODEL ENCODER AND DECODER ########################
class DiffusionEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none', timesteps=100):
        super(DiffusionEncoder, self).__init__()
        
        self.timesteps = timesteps
        self.network = nn.Sequential(
            ConvEncoder(num_inputs, num_channels, embedding_dim, kernel_size, stride, padding, dropout, normalization)
        )
        
        # Noise schedule (simple linear schedule here)
        self.noise_schedule = torch.linspace(1e-4, 0.02, self.timesteps)

    def forward(self, x):
        # Forward diffusion process: Add noise over timesteps
        for t in range(self.timesteps):
            noise = torch.randn_like(x) * self.noise_schedule[t]
            x = x + noise
        
        # Apply the ConvEncoder to the noisy data
        x = self.network(x)
        
        return x
    
    
class DiffusionDecoder(ConvDecoder):
    def __init__(self, *args, **kwargs):
        super(DiffusionDecoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        # Regular decoder process
        x = super(DiffusionDecoder, self).forward(x)
        return x

class DiffusionAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_size, dropout,
                 normalization=None, stride=2, padding=0, timesteps=100, name='Diffusion_AE'):
        super(DiffusionAE, self).__init__(name=name)

        self.encoder = DiffusionEncoder(input_size, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dropout=dropout, normalization=normalization, timesteps=timesteps)
        self.decoder = DiffusionDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride=stride,
                                        padding=padding, dropout=dropout, normalization=normalization)

    def forward(self, x):
        # Pass through the diffusion-based encoder
        x = self.encoder(x)
        
        # Pass through the decoder
        x = self.decoder(x)
        
        return x




######## IMPROVED DIFFUSION MODEL
class AttenDiffusionEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none', timesteps=100, num_heads=4):
        super(AttenDiffusionEncoder, self).__init__()
        
        self.timesteps = timesteps
        self.network = nn.Sequential(
            ConvEncoder(num_inputs, num_channels, embedding_dim, kernel_size, stride, padding, dropout, normalization)
        )
        
        # Noise schedule (simple linear schedule here)
        self.noise_schedule = torch.linspace(1e-4, 0.02, self.timesteps)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Forward diffusion process: Add noise over timesteps
        for t in range(self.timesteps):
            noise = torch.randn_like(x) * self.noise_schedule[t]
            x = x + noise
        
        # Apply the ConvEncoder to the noisy data
        x = self.network(x)
        
        # Apply attention to the encoded features
        # x shape: [batch_size, embedding_dim, seq_len] -> [seq_len, batch_size, embedding_dim] for attention
        x = x.permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(attn_output + x)
        
        # Reshape back to [batch_size, embedding_dim, seq_len]
        x = x.permute(1, 2, 0)
        
        return x



## Improved attention diffusion model 
class AttenDiffusionAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_size, dropout,
                 normalization=None, stride=2, padding=0, timesteps=100, name='Diffusion_AE'):
        super(AttenDiffusionAE, self).__init__(name=name)

        self.encoder = AttenDiffusionEncoder(input_size, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dropout=dropout, normalization=normalization, timesteps=timesteps)
        self.decoder = DiffusionDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride=stride,
                                        padding=padding, dropout=dropout, normalization=normalization)

    def forward(self, x):
        # Pass through the diffusion-based encoder
        x = self.encoder(x)
        
        # Pass through the decoder
        x = self.decoder(x)
        
        return x
    
    
from timm.models.layers import trunc_normal_

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = x_fft * freq_mask.to(x.device)

        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high
        x_weighted += x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(x_in.dtype).view(B, N, C)
        return x


class TimeAttentionCNNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none', num_heads=4, seq_len=100):
        super(TimeAttentionCNNEncoder, self).__init__()

        # CNN-based feature extraction
        self.cnn_encoder = ConvEncoder(num_inputs, num_channels, embedding_dim, kernel_size, stride, padding, dropout, normalization)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, embedding_dim))

        # Multi-head self-attention for temporal attention
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Spectral Block for frequency attention
        self.spectral_block = Adaptive_Spectral_Block(embedding_dim)

    def forward(self, x):
        # CNN Encoder
        x = self.cnn_encoder(x)

        # Reshape for attention
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, embedding_dim]

        # Add positional encoding
        x = x + self.positional_encoding[:x.size(0), :]

        # Temporal Attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(attn_output + x)

        # Spectral Block for adaptive frequency attention
        x = self.spectral_block(x)

        # Reshape back
        x = x.permute(1, 2, 0)  # [batch_size, embedding_dim, seq_len]
        return x


class TimeAttentionCNNAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_size, dropout,
                 normalization=None, stride=2, padding=0, name='TimeAttentionCNN_AE', num_heads=4):
        super(TimeAttentionCNNAE, self).__init__(name=name)

        self.encoder = TimeAttentionCNNEncoder(input_size, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride,
                                               padding=padding, dropout=dropout, normalization=normalization, num_heads=num_heads)
        self.decoder = ConvDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)
        
    def forward(self, x):
        # Pass through the encoder with time attention
        x = self.encoder(x)
        
        # Pass through the decoder
        x = self.decoder(x)
        
        return x
    
    
    
    
    
    
    
## Transformer Autoencoder
# Define Transformer Encoder Block
# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, embedding_dim, seq_len, num_heads=8, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        # Ensure embedding_dim is divisible by num_heads
        if embedding_dim % num_heads != 0:
            embedding_dim = (embedding_dim // num_heads) * num_heads  # Adjust to nearest multiple of num_heads

        # Input projection from input_dim to embedding_dim
        self.input_projection = nn.Linear(input_dim, embedding_dim)

        # Positional encoding for the sequence
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embedding_dim))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Mean Pooling layer to reduce sequence length to match CNN encoder output
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)  # Keep 1 for output consistency

    def forward(self, x):
        # Ensure input is a PyTorch tensor
        # Project input to embedding space
        x = self.input_projection(x)  # Output: [batch_size, seq_len, embedding_dim]

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Output: [batch_size, seq_len, embedding_dim]

        # Permute for mean pooling: [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim, seq_len]
        x = x.permute(0, 2, 1)

        # Apply mean pooling to reduce sequence length to 1
        x = self.mean_pooling(x)  # Output: [batch_size, embedding_dim, 1]


        return x
    

class TransformerAE(nn.Module):
    def __init__(self, input_size, embedding_dim, num_heads, num_filters, seq_len, kernel_size, stride=2, padding=0, dropout=0.2, dim_feedforward=512):
        super(TransformerAE, self).__init__()

        # Transformer-based encoder
        self.encoder = TransformerEncoderBlock(
            input_dim=input_size,
            embedding_dim=embedding_dim,
            seq_len = seq_len,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


        # ConvDecoder
        self.decoder = ConvDecoder(
            embedding_dim=embedding_dim,
            num_channels=num_filters,  # Reverse of encoder filters
            seq_len=seq_len,
            out_dimension=input_size,  # Output should match the input feature size
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout
        )

    def forward(self, x):
        # Encode the input sequence
        encoded_output = self.encoder(x)
        

        # Decode the projected latent representation
        decoded_output = self.decoder(encoded_output)

        return decoded_output
    
    
    

