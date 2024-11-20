# Importing required libraries and modules
import torch  # Core library for deep learning
import torch.nn as nn  # Provides building blocks for neural networks
import torch.nn.functional as F  # Commonly used functions like activation functions
import numpy as np  # Library for numerical operations
import pandas as pd  # For handling data in tabular format
import matplotlib.pyplot as plt  # For visualizing data
from timm import create_model, list_models  # Pretrained models and utilities
from types import SimpleNamespace  # Used to create lightweight objects with attribute-style access
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
)  # Transformer models, tokenization, and learning rate scheduler
import albumentations as A  # Library for augmenting image data
from albumentations.pytorch import ToTensorV2  # Converts images to PyTorch tensors
from PIL import Image  # Library for image processing
from pathlib import Path  # For working with file paths
from sklearn.model_selection import train_test_split  # Utility to split data into train/validation sets
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from tqdm.auto import tqdm  # Progress bar for loops
import gc  # Garbage collection
import json  # To handle JSON files

# Disable parallelism in tokenizers for efficiency
%env TOKENIZERS_PARALLELISM = false

# Define image transformations for data augmentation
sample_tfms = [
    A.HorizontalFlip(),  # Randomly flip images horizontally
    A.RandomBrightnessContrast(),  # Adjust brightness and contrast
    A.ColorJitter(),  # Randomly change brightness, contrast, and saturation
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),  # Shift, scale, and rotate
    A.HueSaturationValue(p=0.3),  # Randomly change hue, saturation, and value
]

# Transformations for training images
train_tfms = A.Compose([
    *sample_tfms,  # Include augmentation transformations
    A.Resize(224, 224),  # Resize images to 224x224
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),  # Normalize pixel values
    ToTensorV2(),  # Convert to PyTorch tensors
])

# Transformations for validation images
valid_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
    ToTensorV2(),
])

# Load a pre-trained GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token
# Define a custom dataset class for loading and processing data
class Dataset:
    def __init__(self, df, tfms):
        """
        Args:
        - df: DataFrame containing image paths and captions
        - tfms: Transformations to be applied to images
        """
        self.df = df
        self.tfms = tfms
    def __len__(self):
        return len(self.df)  # Number of samples in the dataset
    def __getitem__(self,idx):
        """
        Load and preprocess a single data sample.
        Args:
        - idx: Index of the data sample
        Returns:
        - image: Transformed image tensor
        - input_ids: Tokenized input IDs for the caption
        - labels: Labels for language modeling
        """
        sample = self.df.iloc[idx,:]
        image = sample['image']
        caption = sample['caption']
        # Open and process the image
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']
        # Process caption
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:] # Shift labels for language modeling

        return image,input_ids,labels

# COCO 2017
base_path = Path('/kaggle/input/coco-2017-dataset/coco2017')
annot = base_path / 'annotations' / 'captions_train2017.json'
with open(annot, 'r') as f:
    data = json.load(f)['annotations']

# Prepare a DataFrame of image paths and captions
samples = [[f'{sample["image_id"]:012d}.jpg', sample['caption']] for sample in data]
df = pd.DataFrame(samples, columns=['image', 'caption'])
df['image'] = df['image'].apply(lambda x: base_path / 'train2017' / x)
df = df.sample(150_000).reset_index(drop=True)  # Sample and reset index

# Visualize some samples from the dataset
sampled_df = df.sample(n=20)
fig, axs = plt.subplots(10, 2, figsize=(20, 30))
for i, row in enumerate(sampled_df.iterrows()):
    ax = axs[i // 2, i % 2]
    image_path = row[1]['image']
    caption = row[1]['caption']
    image = Image.open(image_path)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(caption)
plt.tight_layout()
plt.show()

# flickr30k
"""
base_path = Path('/kaggle/input/flickr30k/flickr30k_images')
df = pd.read_csv('/kaggle/input/flickr30k/captions.txt',delimiter=',')
df.rename({'image_name':'image','comment': 'caption'},inplace=True,axis=1)
df['image'] = df['image'].map(lambda x:base_path / x.strip())
df['caption'] = df['caption'].map(lambda x:x.strip().lower())
df.head()
"""
# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df,test_size=0.1)
train_df.reset_index(drop=True,inplace=True)
val_df.reset_index(drop=True,inplace=True)
print(len(train_df),len(val_df))

# Create Dataset objects for training and validation
train_ds = Dataset(train_df,train_tfms)
val_ds = Dataset(val_df,valid_tfms)

#Pad according to the longest sequence in the batch
def collate_fn(batch):
    """
    Custom function to collate and pad batch samples.
    Args:
    - batch: List of samples
    Returns:
    - image: Batch of images
    - input_ids: Padded input IDs
    - labels: Padded labels
    """
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    # Stack and pad tensors
    image = torch.stack(image,dim=0)
    input_ids = tokenizer.pad({'input_ids': input_ids}, padding='longest', return_tensors='pt')['input_ids']
    labels = tokenizer.pad({'input_ids': labels}, padding='longest', return_tensors='pt')['input_ids']
    # Mask padding tokens in labels
    mask = (input_ids!=tokenizer.pad_token_id).long()
    labels[mask==0]=-100
    return image, input_ids, labels

# DataLoader for training
dl = torch.utils.data.DataLoader(train_ds,shuffle=True,batch_size=2,collate_fn=collate_fn)
_,c,l = next(iter(dl))
print(c[0])
print(l[0])

# This class implements self-attention, a critical component of the Transformer architecture. 
# It allows a model to attend to different parts of the input sequence to understand context and relationships
class GPT2Attention(nn.Module):
    def __init__(self,config):
        """
        Initializes the GPT2Attention module.

        Args:
            config: Configuration object containing:
                - embed_dim (int): Embedding dimension of the model.
                - num_heads (int): Number of attention heads.
                - seq_len (int): Sequence length for positional encoding.
                - attention_dropout (float): Dropout rate for attention weights.
                - residual_dropout (float): Dropout rate for the residual connections.

        Attributes:
            embed_dim: Total embedding dimension.
            n_heads: Number of attention heads.
            head_size: Dimension of each head (embed_dim / n_heads).
            seq_len: Maximum sequence length.
            c_attn: Linear layer to project input embeddings into query, key, and value vectors.
            scale: Scaling factor for attention scores (1 / sqrt(head_size)).
            mask: Lower triangular mask to enforce causal attention.
            c_proj: Linear layer for projecting output back to embedding dimension.
            attn_dropout: Dropout applied to attention weights.
            resid_dropout: Dropout applied to residual connections.
        """
        # Initializes the parent nn.Module class to ensure proper functionality of PyTorch's module framework (e.g., parameter registration and hooks).
        super().__init__()
        # Stores the embedding dimension of the model from the configuration. This is the size of the input feature vector that will be processed by the attention mechanism.
        self.embed_dim = config.embed_dim
        # Stores the number of attention heads from the configuration. Multi-head attention splits the input into multiple heads for parallel processing, 
        # improving the model's ability to capture diverse relationships
        self.n_heads = config.num_heads
        # Ensures that the embedding dimension is evenly divisible by the number of attention heads. Each attention head will process a subset of the embedding dimension (head_size). 
        # This assertion prevents dimension mismatch issues
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        # Calculates the size of each attention head
        self.head_size = self.embed_dim // self.n_heads
        # Stores the sequence length of the input from the configuration. Required for constructing the attention mask to prevent attending to future tokens in the sequence.
        self.seq_len = config.seq_len
        # Defines a linear layer to project the input into three separate spaces (queries, keys, and values). The output dimension is 3 × (head_size × n_heads) to simultaneously compute 
        # queries, keys, and values for the attention mechanism.
        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=True)
        # Precomputes the scaling factor for the dot-product attention mechanism. Scaling the dot product of queries and keys by the inverse square root 
        # of the head size stabilizes the softmax output and improves training.
        self.scale = self.head_size ** -0.5
        # Registers a lower triangular mask as a persistent buffer. Prevents attending to future tokens in the sequence during training by masking out 
        # upper triangular elements (causal attention). This ensures autoregressive behavior
        self.register_buffer('mask',torch.tril(torch.ones(1,1,self.seq_len,self.seq_len)))
        
        # Defines a linear layer to project the concatenated output of attention heads back to the original embedding space. Combines the information from all attention heads 
        # into a single vector of the same dimension as the input
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        # Adds a dropout layer for the attention scores. Prevents overfitting during training by randomly dropping some connections during attention computation.
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        # Adds a dropout layer for the final output of the attention block. Reduces overfitting by regularizing the residual connection output.
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        b,t,c = x.shape
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        # Project input to query, key, and value tensors
        q,k,v = self.c_attn(x).chunk(3,dim=-1)
        # Reshape for multi-head attention
        q = q.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        
        # Compute scaled dot-product attention
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf'))
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        # Compute attention output
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        # Project output and apply dropout
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out

class GPT2CrossAttention(nn.Module):
    def __init__(self,config):
        """
        Initializes the GPT2CrossAttention module.

        Args:
            config: Configuration object containing:
                - embed_dim (int): Embedding dimension of the model.
                - num_heads (int): Number of attention heads.
                - seq_len (int): Sequence length for positional encoding.
                - attention_dropout (float): Dropout rate for attention weights.
                - residual_dropout (float): Dropout rate for the residual connections.

        Attributes:
            q, k, v: Linear layers for generating query, key, and value vectors.
            scale: Scaling factor for attention scores (1 / sqrt(head_size)).
            c_proj: Linear layer for projecting output back to embedding dimension.
            attn_dropout: Dropout applied to attention weights.
            resid_dropout: Dropout applied to residual connections.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.q = nn.Linear(self.embed_dim,self.embed_dim)
        self.k = nn.Linear(self.embed_dim,self.embed_dim)
        self.v = nn.Linear(self.embed_dim,self.embed_dim)

        self.scale = self.head_size ** -0.5
        
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initializes weights for linear layers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        
    def forward(self, q,k,v):
        """
        Forward pass for cross-attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, query_seq_len, embed_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, key_seq_len, embed_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, value_seq_len, embed_dim).

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, query_seq_len, embed_dim).
        """
        b,t,c = q.shape
        
        # Project query, key, and value
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        # Reshape for multi-head attention
        q = q.view(b,q.size(1),self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,k.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,v.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        
        # Compute scaled dot-product attention
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        # Compute attention output
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        # Project output and apply dropout
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out

class GPT2MLP(nn.Module):
    """
    Implements a Multi-Layer Perceptron (MLP) used in the GPT-2 model for feed-forward transformations.
    The MLP consists of two linear layers with a GELU activation in between, followed by dropout for regularization.
    It expands the feature dimensions in the hidden layer and projects back to the original size.
    """
    def __init__(self, config):
        super().__init__()
        # Embedding dimension of the input features
        self.embed_dim = config.embed_dim
        # Ratio to expand the input dimension in the hidden layer
        self.mlp_ratio = config.mlp_ratio
        # Dropout rate for regularization in the MLP
        self.mlp_dropout = config.mlp_dropout

        # Linear layer to project input to a larger dimension (embed_dim * mlp_ratio)
        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        # Linear layer to project the expanded dimension back to the original embedding size
        self.c_proj = nn.Linear(self.embed_dim * self.mlp_ratio, self.embed_dim)
        # GELU activation function for smooth non-linearity
        self.act = nn.GELU()
        # Dropout layer to randomly zero out activations for regularization
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Output tensor of the same shape as input (batch_size, seq_len, embed_dim).
        """
        # First linear transformation to expand the input features
        x = self.c_fc(x)
        # Apply the GELU activation function
        x = self.act(x)
        # Project the expanded features back to the original embedding size
        x = self.c_proj(x)
        # Apply dropout for regularization
        x = self.dropout(x)
        # Return the transformed features
        return x

class GPT2Block(nn.Module):
    """
    Implements a single block of the GPT-2 architecture, consisting of:
    - A self-attention mechanism
    - Cross-attention for encoder-decoder interaction
    - A feed-forward MLP
    - Layer normalization after each sub-layer
    Each sub-layer is followed by residual connections.
    """
    def __init__(self, config):
        super().__init__()
        # Embedding dimension of the input features
        self.embed_dim = config.embed_dim

        # Layer normalization applied before the self-attention mechanism
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        # Self-attention mechanism
        self.attn = GPT2Attention(config)

        # Layer normalization applied before the cross-attention mechanism
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        # Cross-attention mechanism for encoder-decoder interaction
        self.cross_attn = GPT2CrossAttention(config)

        # Layer normalization applied before the feed-forward MLP
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        # Feed-forward MLP block
        self.mlp = GPT2MLP(config)

    def forward(self, x, enc_out):
        """
        Forward pass through the GPT-2 block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            enc_out (Tensor): Encoder output tensor for cross-attention, shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Apply layer normalization and self-attention, followed by a residual connection
        x = x + self.attn(self.ln_1(x))
        # Apply layer normalization and cross-attention with encoder output, followed by a residual connection
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
        # Apply layer normalization and the MLP, followed by a residual connection
        x = x + self.mlp(self.ln_3(x))
        # Return the final output
        return x


class VisionGPT2Model(nn.Module):
    """
    Combines a Vision Transformer (ViT) with GPT-2 to create a vision-language model.
    The model integrates image patch embeddings from ViT with GPT-2's transformer layers for sequence modeling.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Load a pre-trained Vision Transformer (ViT) model with no output classification layer
        vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

        # Patch embedding module from ViT, which divides the input image into patches and embeds them
        self.patch_embed = vit.patch_embed
        # Total number of patches generated by the ViT patch embedding
        num_patches = self.patch_embed.num_patches

        # Class token for ViT, used to aggregate global information from all patches
        self.cls_token = vit.cls_token

        # Compute the total embedding length, including class token and positional embeddings
        embed_len = num_patches + vit.num_prefix_tokens
        # Positional embedding for the patches and class token
        self.pos_embed = vit.pos_embed
        # Dropout for positional embeddings to reduce overfitting
        self.pos_drop = nn.Dropout(p=0.)

        # Subset of ViT transformer blocks up to the specified depth in the configuration
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])

        # GPT-2 style transformer, adapted for vision-language tasks
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings for input vocabulary
            wte=nn.Embedding(config.vocab_size, config.embed_dim),
            # Positional embeddings for sequence tokens
            wpe=nn.Embedding(config.seq_len, config.embed_dim),
            # Dropout for embeddings
            drop=nn.Dropout(config.emb_dropout),
            # Transformer blocks for processing sequences
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            # Final layer normalization
            ln_f=nn.LayerNorm(config.embed_dim)
        ))

        # Linear layer to project transformer output to vocabulary logits
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Share weights between the embedding layer and the language modeling head
        self.transformer.wte.weight = self.lm_head.weight

    def _pos_embed(self, x):
        """
        Add positional embeddings to input patches and prepend the class token.

        Args:
            x (Tensor): Patch embeddings of shape (batch_size, num_patches, embed_dim).

        Returns:
            Tensor: Input with positional embeddings and class token, shape (batch_size, embed_len, embed_dim).
        """
        # Retrieve positional embeddings
        pos_embed = self.pos_embed
        # Prepend class token to patch embeddings
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # Add positional embeddings
        x = x + pos_embed
        # Apply dropout
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable=False):
        """
        Toggle trainability of pretrained layers (ViT and GPT-2).

        Args:
            trainable (bool): Whether to make layers trainable (True) or freeze them (False).
        """
        # List of all layers in the model
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]

        # Add GPT-2 transformer layers to the list
        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        for l in gpt_layers:
            layers.extend(l)

        # Set the requires_grad property for each layer's parameters
        for layer in layers:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        # Calculate and print the number of frozen parameters
        total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'{total_frozen_params=}')

    def unfreeze_gpt_layers(self):
        """
        Unfreeze all GPT-2 transformer layers, enabling them to be trainable.
        """
        # Flatten all GPT-2 layers for easy iteration
        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        flatten = []
        for l in gpt_layers:
            flatten.extend(l)

        # Set the requires_grad property for each parameter in GPT-2 layers
        for layer in flatten:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True

    # The @classmethod decorator in Python is used to define a method that belongs to the class itself rather than to a specific instance of the class. 
    # This means the method can be called on the class directly without needing to instantiate an object of the class.  
    @classmethod
    def from_pretrained(cls, config):
        """
        Load a pre-trained VisionGPT2Model by combining ViT and GPT-2 pretrained weights.

        Args:
            config (object): Configuration object containing model parameters.

        Returns:
            VisionGPT2Model: A VisionGPT2Model instance initialized with pretrained weights.
        """
        # Initialize a new model instance
        model = cls(config)
        sd = model.state_dict()  # Model's current state dictionary
        keys = sd.keys()  # All keys in the state dictionary

        # Define patterns to identify which weights to ignore
        ignore_matches = ['blocks.', 'cross_attn.', 'ln_3', 'cls_token', 'pos_embed', 'patch_embed.', '.attn.mask']

        # Separate ViT-specific keys
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        # Remaining keys are considered GPT-specific
        gpt_keys = [key for key in keys if key not in vit_keys]

        # Load pretrained GPT-2 weights
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()  # GPT-2's state dictionary
        hf_keys = sd_hf.keys()  # All GPT-2 keys

        # Filter GPT-2 keys to exclude unnecessary masked bias and bias keys
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]

        # Define keys that need transposing for weight shape compatibility
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Copy relevant weights from GPT-2 to the model
        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue  # Skip ignored keys
            if any(k.endswith(w) for w in transposed):  # Transpose weights if needed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:  # Direct copy for matching shapes
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # Load updated state dictionary into the model
        model.load_state_dict(sd)

        return model

    def forward(self, image, input_ids, labels=None):
        """
        Forward pass for vision-language processing.

        Args:
            image (Tensor): Input image tensor.
            input_ids (Tensor): Sequence of token IDs.
            labels (Tensor, optional): Ground truth labels for loss computation.

        Returns:
            Tensor: Loss value if labels are provided, else logits of the final token.
        """
        # Extract patch embeddings from the input image
        image = self.patch_embed(image)
        image = self._pos_embed(image)

        # Get token embeddings and positional embeddings
        token_embeddings = self.transformer.wte(input_ids)  # Token embeddings
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)  # Positional embeddings
        input_ids = self.transformer.drop(token_embeddings + positional_embeddings)  # Combine with dropout

        # Process both image and input tokens through transformer layers
        for i in range(self.config.depth):
            image = self.blocks[i](image)  # Process image embeddings through ViT blocks
            input_ids = self.transformer.h[i](input_ids, image)  # Fuse image and token embeddings

        # Apply final layer normalization
        input_ids = self.transformer.ln_f(input_ids)

        # Compute loss if labels are provided
        if labels is not None:
            lm_logits = self.lm_head(input_ids)  # Project to vocabulary space
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss

        # Return logits of the last token if no labels are provided
        lm_logits = self.lm_head(input_ids[:, [-1], :])
        return lm_logits

    def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False):
        """
        Generate a sequence of tokens using autoregressive decoding.

        Args:
            image (Tensor): Input image tensor.
            sequence (Tensor): Initial sequence of token IDs.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature to control randomness.
            deterministic (bool): Use deterministic (greedy) sampling if True.

        Returns:
            Tensor: Generated token sequence.
        """
        for _ in range(max_tokens):
            # Predict the next token
            out = self(image, sequence)
            out = out[:, -1, :] / temperature  # Normalize logits by temperature
            probs = F.softmax(out, dim=-1)  # Convert logits to probabilities

            # Select the next token deterministically or probabilistically
            if deterministic:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the sequence
            sequence = torch.cat([sequence, next_token], dim=1)

            # Stop generation if the end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

        return sequence.cpu().flatten()

class Trainer:
    def __init__(self, model_config, train_config, dls):
        """
        Initialize the Trainer with configurations and data loaders.

        Args:
            model_config (SimpleNamespace): Configuration for the VisionGPT2Model.
            train_config (SimpleNamespace): Training configurations, including epochs, learning rate, etc.
            dls (tuple): A tuple containing training and validation dataloaders.

        Attributes:
            model: The VisionGPT2Model initialized with pretrained weights.
            tokenizer: GPT2 tokenizer for text processing.
            scaler: Gradient scaler for mixed-precision training.
            train_dl, val_dl: Training and validation dataloaders.
            optim: Adam optimizer for the model.
            sched: OneCycleLR scheduler for learning rate scheduling.
            metrics: DataFrame to track training and validation losses and perplexities.
            gen_tfms: Transformations for image preprocessing during caption generation.
        """
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        # Load and prepare the model
        self.model = VisionGPT2Model.from_pretrained(model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)

        print(f'Trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        # Prepare tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Gradient scaler for mixed precision
        self.scaler = GradScaler()

        # Data loaders
        self.train_dl, self.val_dl = dls

        # Optimizer and scheduler
        total_steps = len(self.train_dl)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.0)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )

        # Metrics DataFrame
        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity', 'val_loss', 'val_perplexity']] = None

        # Image preprocessing for caption generation
        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])

    def save_model(self):
        """Save the current model state to a file."""
        self.train_config.model_path.mkdir(exist_ok=True)
        sd = self.model.state_dict()
        torch.save(sd, self.train_config.model_path / 'captioner.pt')

    def load_best_model(self):
        """Load the best saved model from file."""
        sd = torch.load(self.train_config.model_path / 'captioner.pt')
        self.model.load_state_dict(sd)

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.
        """
        prog = tqdm(self.train_dl, total=len(self.train_dl))
        running_loss = 0.0

        for image, input_ids, labels in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)

                # Backpropagation and optimization
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()
                prog.set_description(f'Train loss: {loss.item():.3f}')

            # Clean up to save memory
            del image, input_ids, labels, loss

        # Compute average loss and perplexity
        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)
        self.metrics.loc[epoch, ['train_loss', 'train_perplexity']] = (train_loss, train_pxp)

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        """
        Validate the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Validation perplexity.
        """
        prog = tqdm(self.val_dl, total=len(self.val_dl))
        running_loss = 0.0

        for image, input_ids, labels in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()
                prog.set_description(f'Valid loss: {loss.item():.3f}')

            # Clean up to save memory
            del image, input_ids, labels, loss

        # Compute average loss and perplexity
        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)
        self.metrics.loc[epoch, ['val_loss', 'val_perplexity']] = (val_loss, val_pxp)
        return val_pxp

    def clean(self):
        """Perform garbage collection and free CUDA memory."""
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self):
        """
        Train and validate the model for multiple epochs.

        Returns:
            dict: Best perplexity and corresponding epoch.
        """
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))

        for epoch in prog:
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('Unfreezing GPT2 entirely...')

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            # Training phase
            self.model.train()
            prog.set_description('Training')
            self.train_one_epoch(epoch)
            self.clean()

            # Validation phase
            self.model.eval()
            prog.set_description('Validating')
            pxp = self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

            # Save the best model
            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('Saving best model...')
                self.save_model()

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False):
        """
        Generate a caption for the given image.

        Args:
            image (str): Path to the input image.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            deterministic (bool): If True, use deterministic generation.

        Returns:
            str: Generated caption.
        """
        self.model.eval()

        # Preprocess the image
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)

        # Prepare initial sequence
        sequence = torch.ones(1, 1).to(device=self.device).long() * self.tokenizer.bos_token_id

        # Generate caption
        caption = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        caption = self.tokenizer.decode(caption.numpy(), skip_special_tokens=True)

        return caption


# Define model configuration using SimpleNamespace for easy attribute access
model_config = SimpleNamespace(
    vocab_size=50_257,            # Size of the vocabulary (e.g., GPT-2's vocabulary size)
    embed_dim=768,                # Embedding dimension for each token
    num_heads=12,                 # Number of attention heads in each transformer block
    seq_len=1024,                 # Maximum sequence length supported by the model
    depth=12,                     # Number of transformer layers in the model
    attention_dropout=0.1,        # Dropout rate for the attention mechanism
    residual_dropout=0.1,         # Dropout rate for residual connections
    mlp_ratio=4,                  # Ratio for the feed-forward layer's hidden size to input size
    mlp_dropout=0.1,              # Dropout rate in the feed-forward layer
    emb_dropout=0.1               # Dropout rate for token embeddings
)

# Define training configuration using SimpleNamespace
train_config = SimpleNamespace(
    epochs=5,                     # Total number of training epochs
    freeze_epochs_gpt=1,          # Number of epochs to freeze GPT layers during training
    freeze_epochs_all=2,          # Number of epochs to freeze all pretrained layers
    lr=1e-4,                      # Initial learning rate for training
    device='cuda',                # Device to use for training ('cuda' for GPU or 'cpu' for CPU)
    model_path=Path('captioner'), # Path to save the trained model
    batch_size=32                 # Batch size for training and validation
)

# Create DataLoader for the training dataset
train_dl = torch.utils.data.DataLoader(
    train_ds,                      # Training dataset
    batch_size=train_config.batch_size, # Batch size for training
    shuffle=True,                  # Shuffle data at every epoch
    pin_memory=True,               # Pin memory for faster data transfer to GPU
    num_workers=2,                 # Number of workers for data loading
    persistent_workers=True,       # Keep workers alive across epochs for efficiency
    collate_fn=collate_fn          # Custom function to process and batch data
)

# Create DataLoader for the validation dataset
val_dl = torch.utils.data.DataLoader(
    val_ds,                        # Validation dataset
    batch_size=train_config.batch_size, # Batch size for validation
    shuffle=False,                 # Do not shuffle validation data
    pin_memory=True,               # Pin memory for faster data transfer to GPU
    num_workers=2,                 # Number of workers for data loading
    persistent_workers=True,       # Keep workers alive across epochs for efficiency
    collate_fn=collate_fn          # Custom function to process and batch data
)

# Initialize the Trainer class with model configuration, training configuration, and DataLoaders
trainer = Trainer(model_config, train_config, (train_dl, val_dl))

# Train the model and fit it on the dataset
trainer.fit()

# View the metrics collected during training and validation
trainer.metrics

# Plot training and validation loss
plt.plot(trainer.metrics['train_loss'], color='red', label='train loss')  # Plot train loss
plt.plot(trainer.metrics['val_loss'], color='orange', label='valid loss')  # Plot validation loss
plt.title('Loss, lower=better')  # Title for the plot
plt.legend()  # Add legend to differentiate curves
plt.show()  # Display the plot

# Plot training and validation perplexity
plt.plot(trainer.metrics['train_perplexity'], color='blue', label='train perplexity')  # Train perplexity
plt.plot(trainer.metrics['val_perplexity'], color='lightblue', label='valid perplexity')  # Validation perplexity
plt.title('Perplexity, lower=better')  # Title for the plot
plt.legend()  # Add legend to differentiate curves
plt.show()  # Display the plot

# Load the best model based on validation performance
trainer.load_best_model()

# Test the model by generating captions for random samples from the validation dataset
for i in range(50):  # Generate captions for 50 samples
    det = False  # Deterministic generation flag, initially set to False
    test = val_df.sample(n=1).values[0]  # Randomly sample one validation data point
    test_img, test_caption = test[0], test[1]  # Extract image path and actual caption

    # Display the test image
    plt.imshow(Image.open(test_img).convert('RGB'))  # Open and convert the image to RGB format

    # Generate a random temperature for diversity in generation
    t = np.random.uniform(0.5, 1.5)

    # Use deterministic generation for the last 10 samples
    if i > 40:
        det = True

    # Generate a caption for the test image
    gen_caption = trainer.generate_caption(test_img, temperature=t, deterministic=det)

    # Display the actual and generated captions along with generation parameters
    plt.title(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {t} deterministic generation: {det}")
    plt.axis('off')  # Remove axis for better visual appeal
    plt.show()  # Show the image and captions

