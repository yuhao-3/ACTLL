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
    
    
    
## Time attention CNN encoder for Time Series
class TimeAttentionCNNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none', num_heads=4):
        super(TimeAttentionCNNEncoder, self).__init__()
        
        # CNN-based feature extraction
        self.cnn_encoder = ConvEncoder(num_inputs, num_channels, embedding_dim, kernel_size, stride, padding, dropout, normalization)
        
        # Multi-head self-attention for temporal attention
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        # Extract features with the CNN encoder
        x = self.cnn_encoder(x)
        
        # Reshape the data for attention: [batch_size, embedding_dim, seq_len] -> [seq_len, batch_size, embedding_dim]
        x = x.permute(2, 0, 1)
        
        # Apply attention to time steps
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection and normalization
        x = self.layer_norm(attn_output + x)
        
        # Reshape back to original dimensions: [seq_len, batch_size, embedding_dim] -> [batch_size, embedding_dim, seq_len]
        x = x.permute(1, 2, 0)
        
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
    
    
# class TransformerDecoderBlock(nn.Module):
#     def __init__(self, embedding_dim, output_dim, seq_len, num_heads=8, dim_feedforward=512, dropout=0.1):
#         super(TransformerDecoderBlock, self).__init__()

#         # Transformer Decoder Layer
#         decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

#         # Positional encoding for the decoder
#         self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embedding_dim))

#         # Output projection from embedding_dim back to input_dim
#         self.output_projection = nn.Linear(embedding_dim, output_dim)

#         # Projection to match sequence length
#         self.projection_to_seq_len = nn.Linear(1, seq_len)

#     def forward(self, latent_seq, memory_seq=None):
#         # If memory_seq is None, use latent_seq as both query and memory
#         if memory_seq is None:
#             memory_seq = latent_seq

#         # Latent sequence comes in with shape [batch_size, embedding_dim]
#         if len(latent_seq.shape) == 2:  # Ensure latent_seq is 3D
#             latent_seq = latent_seq.unsqueeze(1)  # Add sequence length: [batch_size, 1, embedding_dim]

#         # Add positional encoding
#         latent_seq = latent_seq + self.positional_encoding[:, :latent_seq.size(1), :]

#         # Decode the latent representation
#         x = self.transformer_decoder(latent_seq, memory_seq)

#         # Project back to original input dimensions
#         x = self.output_projection(x)

#         # Ensure the final projection matches the original sequence length
#         x = self.projection_to_seq_len(x.transpose(1, 2)).transpose(1, 2)

#         return x  # Return shape: [batch_size, output_dim, seq_len]
    

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
    
    
    
######################## Inception Module ######################
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9], bottleneck_channels=32, residual=True):
        super(InceptionBlock, self).__init__()
        self.residual = residual

        # Ensure that in_channels is an integer, since input_size is a single integer
        assert isinstance(in_channels, int), "in_channels should be an integer"
        
        # Bottleneck layer to reduce input channels before applying multiple kernels
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1) if in_channels > 1 else nn.Identity()

        # If out_channels is an integer, apply the same number of filters to each convolution
        if isinstance(out_channels, int):
            out_channels = [out_channels] * len(kernel_sizes)

        # Ensure the length of out_channels matches kernel_sizes
        assert len(out_channels) == len(kernel_sizes), "Length of out_channels must match length of kernel_sizes"

        # Apply convolutions with different kernel sizes, each with its own out_channels
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels[i], kernel_size=k, padding=k // 2) for i, k in enumerate(kernel_sizes)
        ])

        # Pooling layer to combine multi-scale features
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # 1x1 convolution to match the output channels to the sum of all out_channels
        self.convpool = nn.Conv1d(in_channels, sum(out_channels), kernel_size=1)

    def forward(self, x):
        # Apply bottleneck layer
        bottleneck_out = self.bottleneck(x)

        # Apply different convolutions and concatenate the results
        inception_out = torch.cat([conv(bottleneck_out) for conv in self.convs], dim=1)

        # Apply pooling and the 1x1 convolution
        maxpool_out = self.convpool(self.maxpool(x))

        # Add the residual connection if enabled
        if self.residual:
            return inception_out + maxpool_out
        else:
            return inception_out
        
class InceptionTemporalEncoder(nn.Module):
    def __init__(self, input_size, num_filters, embedding_dim, kernel_sizes, num_modules, dropout):
        super(InceptionTemporalEncoder, self).__init__()

        # Ensure input_size is a single integer
        assert isinstance(input_size, int), "input_size should be an integer"
        
        # Handle num_filters as either a list or a single integer
        if isinstance(num_filters, int):
            num_filters_list = [num_filters] * num_modules  # Repeat the same filter size for all blocks
        else:
            assert isinstance(num_filters, list), "num_filters should be either an integer or a list"
            num_filters_list = num_filters

        # Initialize Inception blocks
        self.inception_blocks = nn.ModuleList([
            InceptionBlock(input_size if i == 0 else num_filters_list[i - 1], num_filters_list[i], kernel_sizes)
            for i in range(num_modules)
        ])

        # Final 1x1 convolution to reduce the output to `batch_size * timestep * 1`
        self.conv1x1 = nn.Conv1d(num_filters_list[-1], 1, kernel_size=1)  # Reduce channels to 1

    def forward(self, x):
        for block in self.inception_blocks:
            x = block(x)
        x = self.conv1x1(x)  # Reduce channel dimension to 1
        return x
    
    
    
class InceptionTemporalDecoder(nn.Module):
    def __init__(self, embedding_dim, num_filters, seq_len, output_size, kernel_size=3, stride=2, padding=1, dropout=0.2):
        super(InceptionTemporalDecoder, self).__init__()

        # Reverse the filter list for decoding
        num_filters = num_filters[::-1]

        # Calculate the compressed length for upsampling
        self.compressed_len = seq_len // (2 ** len(num_filters))  # Assuming stride 2 across layers

        # Upsample the latent vector to match the reduced sequence length
        self.upsample = nn.Linear(embedding_dim, self.compressed_len * num_filters[0])

        # Create a list of transposed convolution layers and dropout layers
        layers = []
        for i in range(len(num_filters)):
            in_channels = num_filters[i - 1] if i > 0 else num_filters[0]
            out_channels = num_filters[i]
            
            # Transposed convolution
            layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            
            # Apply dropout layer after each ConvTranspose1d
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Final 1x1 convolution to restore the original input size
        self.conv1x1 = nn.Conv1d(num_filters[-1], output_size, kernel_size=1)

    def forward(self, x):
        # Upsample the latent vector
        x = self.upsample(x).view(x.size(0), -1, self.compressed_len)
        
        # Pass through transposed convolution layers
        x = self.network(x)
        
        # Apply 1x1 convolution to match the output dimension
        x = self.conv1x1(x)
        return x 
    
    
    
class InceptionTemporalAE(nn.Module):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_sizes=[3, 5, 7], num_modules=4, kernel_size=3, stride=2, padding=1, dropout=0.2):
        super(InceptionTemporalAE, self).__init__()

        # Encoder: Inception Temporal Encoder
        self.encoder = InceptionTemporalEncoder(input_size, num_filters, embedding_dim, kernel_sizes, num_modules, dropout)

        # Decoder: Inception Temporal Decoder
        self.decoder = InceptionTemporalDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride, padding, dropout)

    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)

        # Decode the latent representation back to the original input shape
        decoded = self.decoder(encoded)

        return decoded