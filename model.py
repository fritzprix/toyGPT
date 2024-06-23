from typing import Any, Tuple, Dict
from transformers import PreTrainedModel
from pydantic import BaseModel
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import math
import lightning as L
from transformers.configuration_utils import PretrainedConfig

# ScaledDotProductAttention is a fundamental component in Transformer architecture.
# It computes attention weights and produces a weighted average of values (v).
class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    # q: query, k: key, v: value, mask: optional mask to exclude certain positions from attention
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Calculate the dot product of q and k, then scale it by the square root of the dimensionality of the keys.
        scaled_qk: torch.Tensor = q @ k.transpose(2, 1) * (1 / math.sqrt(k.size(-1)))
        # If a mask is provided (e.g., for padding or future blinding), apply it to the scaled dot product.
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask.bool().bitwise_not(), float('-inf'))
        
        # Apply softmax to get the attention weights, then multiply with the values.
        attention_weights = torch.softmax(scaled_qk, dim=-1)
        return attention_weights @ v

class MultiHeadAttentionFA(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float32, dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.p_dropout = dropout

        self.attn_proj = torch.nn.Sequential(torch.nn.Linear(d_model, 3 * d_model, device=device, dtype=dtype),
                                             torch.nn.Dropout(dropout))
        
        self.attn_dropout = torch.nn.Dropout(dropout)
        
        self.out_linear = torch.nn.Linear(d_model, d_model)
        self.out_dropout = torch.nn.Dropout(dropout)

        
    def forward(self, input: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if len(input.shape) != 3:
            raise ValueError(f'unsupported tensor shape: {input.shape}, should be form of (B,N,d)')
        
        
        B,n_seq,C = input.size()
        q ,k, v = self.attn_proj(input).split(C, dim=-1)
        
        q = q.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2) # (B, n_h, n_seq, d_h)
        k = k.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2) # (B, n_h, n_seq, d_h)
        v = v.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2)

        sdp_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p_dropout if self.training else 0)

        return self.out_dropout(self.out_linear(sdp_out.transpose(1,2).view(B, n_seq, C)))

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float32, dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head

        self.attn_proj = torch.nn.Sequential(torch.nn.Linear(d_model, 3 * d_model, device=device, dtype=dtype),
                                             torch.nn.Dropout(dropout))
        
        self.attn_dropout = torch.nn.Dropout(dropout)
        
        self.out_linear = torch.nn.Linear(d_model, d_model)
        self.out_dropout = torch.nn.Dropout(dropout)

        
    def forward(self, input: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if len(input.shape) != 3:
            raise ValueError(f'unsupported tensor shape: {input.shape}, should be form of (B,N,d)')
        
        
        B,n_seq,C = input.size()
        q ,k, v = self.attn_proj(input).split(C, dim=-1)
        
        q = q.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2) # (B, n_h, n_seq, d_h)
        k = k.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2) # (B, n_h, n_seq, d_h)
        v = v.view(B, n_seq, self.n_head, C // self.n_head).transpose(1,2)

        scaled_dot_product = q@k.transpose(-1,-2) * (1 / math.sqrt(k.size(-1))) # (B, n_h, n_seq, n_seq)
        if mask is not None:
            # shape of given mask (B, n_seq, n_seq) we have to unsqueeze to get (B, 1, n_seq,n_seq) so it can be broadcast to (B,n_head, n_seq,n_seq)
            scaled_dot_product = scaled_dot_product.masked_fill(mask=mask.unsqueeze(1).bitwise_not(), value=float('-inf'))
        
        sdp_out: torch.Tensor = self.attn_dropout(scaled_dot_product).softmax(dim=-1) @ v # (B, n_h, n_seq, d_h)

        return self.out_dropout(self.out_linear(sdp_out.transpose(1,2).view(B,n_seq, C)))



class MultiHeadAttentionNaive(torch.nn.Module):

    def __init__(self, d_model:int, n_head:int, device=None, dtype: torch.dtype=torch.float32, dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout

        self.qkv_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, 3 * d_model, device=device, dtype=dtype), 
            torch.nn.Dropout(dropout))

        self.attns = torch.nn.ModuleList([ScaledDotProductAttention() for _ in range(n_head)])
        
        self.output_linear = torch.nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_drop = torch.nn.Dropout(dropout)

        
    def forward(self, input: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        if len(input.shape) != 3:
            raise ValueError(f'unsupported tensor shape: {input.shape}, should be form of (B,N,d)')
        
        batch_size,seq_n, d_model = input.shape
        qkv_bundle: torch.Tensor = self.qkv_proj(input)
        q,k,v = qkv_bundle.split(d_model,-1)
        
        q = q.view((batch_size, seq_n, -1, self.d_head)).transpose(1, 2)
        k = k.view((batch_size, seq_n, -1, self.d_head)).transpose(1, 2)
        v = v.view((batch_size, seq_n, -1, self.d_head)).transpose(1, 2)
        # now Q,K,V have shape of (batch_size, seq_n, n_head, d_head)


        attn_output = torch.concat([self.attns[i].forward(q[:,i,:,:].squeeze(1), 
                                            k[:,i,:,:].squeeze(1), 
                                            v[:,i,:,:].squeeze(1),mask=mask) for i in range(self.n_head)],dim=-1)
        
        return self.out_drop(self.output_linear(attn_output))
        


class PositionWiseFeedforward(torch.nn.Module):

    def __init__(self, d_model:int, device, dtype: torch.dtype=torch.float32, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pwff = torch.nn.Sequential(torch.nn.Linear(d_model, d_model * 4, device=device,dtype=dtype), 
                                            torch.nn.GELU(), 
                                            torch.nn.Linear(4* d_model, d_model, device=device, dtype=dtype), 
                                            torch.nn.Dropout(dropout))
        
    def forward(self, input: torch.Tensor)-> torch.Tensor:
        return self.pwff.forward(input)


# The Transformer class defines a single Transformer block, which is composed of a multi-head attention layer
# followed by a position-wise feedforward network.
class TransformerFA(torch.nn.Module):
    # Initialization of the Transformer block with normalization layers, multi-head attention, and feedforward network.
    def __init__(self, n_head, d_model, device, dtype:torch.dtype=torch.float32,dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Normalization layer applied to the input.
        self.input_norm = torch.nn.LayerNorm(d_model, device=device, dtype=dtype)
        # Multi-head attention layer.
        self.mha = MultiHeadAttentionFA(d_model=d_model, n_head=n_head, device=device, dtype=dtype, dropout=dropout)
        # Normalization layer applied after the attention layer.
        self.mha_lnorm = torch.nn.LayerNorm(d_model, device=device,dtype=dtype)
        # Position-wise feedforward network.
        self.pw_ff = PositionWiseFeedforward(d_model=d_model, device=device, dtype=dtype, dropout=dropout)
        # Dropout layer for the residual connections.
        self.residual_dropout = torch.nn.Dropout(dropout)

    # The forward method defines how the input data flows through the Transformer block.
    def forward(self, data:Tuple[torch.Tensor]) -> torch.Tensor:
        # Split the input tuple into the input tensor and the attention mask.
        input, attention_mask = data

        # Apply layer normalization.
        norm_input = self.input_norm.forward(input)
        # Compute the output of the multi-head attention layer, adding the input for residual connection.
        mha_output = self.residual_dropout(input) + self.mha.forward(norm_input, attention_mask)
        # Apply normalization to the output of the attention layer.
        norm_mha_output = self.mha_lnorm(mha_output)
        # Return the output of the feedforward network, again adding the input for residual connection, along with the attention mask.
        return (mha_output + self.pw_ff.forward(norm_mha_output), attention_mask)


def positional_embedding(d_model, max_length, dtype=None, device=None):
        div_even = torch.pow(10000, torch.arange(0, d_model // 2, dtype=dtype, device=device) * 2 / d_model)
        div_odd = torch.pow(10000, torch.arange(0, d_model // 2, dtype=dtype, device=device) * 2 / d_model)
        pos = torch.arange(0, max_length, device=device, dtype=dtype).unsqueeze(-1)
        pe = torch.zeros((max_length, d_model), device=device, dtype=dtype)
        pe[:,0::2] = torch.sin(pos / div_even)
        pe[:,1::2] = torch.cos(pos / div_odd)
        return pe.requires_grad_(False)


class TransformerNaive(torch.nn.Module):

    def __init__(self, n_head, d_model, device, dtype:torch.dtype=torch.float32,dropout:float=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_norm = torch.nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttentionNaive(d_model=d_model, n_head=n_head, device=device, dtype=dtype, dropout=dropout)
        self.mha_lnorm = torch.nn.LayerNorm(d_model, device=device,dtype=dtype)
        self.pw_ff = PositionWiseFeedforward(d_model=d_model, device=device, dtype=dtype, dropout=dropout)
        self.residual_dropout = torch.nn.Dropout(dropout)

    def forward(self, data:Tuple[torch.Tensor]) -> torch.Tensor:
        # Pre-LayerNormalization from GPT-3, (note: Post-LayerNormalization is used for GPT-2 and original paper)
        input, attention_mask = data

        norm_input = self.input_norm.forward(input)
        mha_output = self.residual_dropout(input) + self.mha.forward(norm_input, attention_mask)
        norm_mha_output = self.mha_lnorm(mha_output)
        return (mha_output + self.pw_ff.forward(norm_mha_output), attention_mask)
    

def positional_embedding(d_model, max_length, dtype=None, device=None):
        div_even = torch.pow(10000, torch.arange(0, d_model // 2, dtype=dtype, device=device) * 2 / d_model)
        div_odd = torch.pow(10000, torch.arange(0, d_model // 2, dtype=dtype, device=device) * 2 / d_model)
        pos = torch.arange(0, max_length, device=device, dtype=dtype).unsqueeze(-1)
        pe = torch.zeros((max_length, d_model), device=device, dtype=dtype)
        pe[:,0::2] = torch.sin(pos / div_even)
        pe[:,1::2] = torch.cos(pos / div_odd)
        return pe.requires_grad_(False)


class ToyGPTModelConfig(BaseModel):
    name: str = 'toygpt'
    n_layer: int = 12
    n_head: int = 8
    block_size: int = 512
    n_embed:int = 768

# The ToyGPT class defines the complete model architecture by stacking multiple Transformer blocks and adding the necessary components for language modeling.
class ToyGPT(L.LightningModule):
    # Initialization of the ToyGPT model with parameters like vocabulary size, block size, etc.
    def __init__(self,
                 vocab_size:int, 
                 block_size:int,
                 batch:int,
                 name: str,
                 n_embed:int, n_head:int, n_layer:int, mask_token_id:int = None, pad_token_id:int=None, 
                 device=None, 
                 dtype:torch.dtype=torch.float32, 
                 p_dropout:float=0.1, 
                 lr=2.5e-4, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.01,  *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        # Save hyperparameters for easy access and checkpointing.
        self.save_hyperparameters(ignore=['dtype', 'device'])
        self.name = name
        self.batch = batch
        # Define the linear layer for mapping the output of the transformers to the vocabulary size.
        self.output_linear = torch.nn.Linear(n_embed, vocab_size, device=device, dtype=dtype)
        # Define the embedding layer for converting token IDs to dense vectors.
        self.embedding = torch.nn.Embedding(vocab_size, n_embed, padding_idx=pad_token_id) 
        # Tying the weights of the output linear layer and the embedding layer.
        self.embedding.weight = self.output_linear.weight 
        # Define the positional embeddings for the model.
        self.pos_embedding = positional_embedding(d_model=n_embed, max_length=block_size, device=device, dtype=dtype).unsqueeze(0)
        # Define the dropout layer for embeddings.
        self.embedding_dropout = torch.nn.Dropout(p_dropout)
        # Stack the Transformer blocks to create the model.
        self.transformers = torch.nn.Sequential(*[TransformerFA(n_head=n_head, d_model=n_embed, device=device, dtype=dtype, dropout=p_dropout) for _ in range(n_layer)])
        # Define the loss function, ignoring the padding index in the loss calculation.
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.decay = weight_decay
        self.mask_token_id = mask_token_id
        # Initialize the weights of the model.
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.decay)
        
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=self.eps, total_iters=2000, end_factor=1),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=526800 * 2, eta_min=(self.lr / 10))
        ], milestones=[2000])
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor":"val_loss",
            "name": "CosineWithWarmUp",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        # X should have shape of (B,N)
        X: torch.Tensor = input["input_ids"]
        attention_mask: torch.Tensor = input["attention_mask"]
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        N = X.size(1)
        
        X_wemb = self.embedding(X) + self.pos_embedding[:,:N,:] # word embedding + postion embedding
        hs, _ = self.transformers.forward((X_wemb, attention_mask.bool()))
        return torch.softmax(self.output_linear.forward(hs[:,-1,:]), -1)
    
    def _calculate_clm_loss(self, input: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> STEP_OUTPUT:

        seq_n = input.size(1)

        X_wemb = self.embedding(input) + self.pos_embedding[:,:seq_n,:] # word embedding + postion embedding
        hidden_output, _ = self.transformers.forward((self.embedding_dropout(X_wemb), attention_mask))
        logits = self.output_linear.forward(hidden_output)
        # the sequencess of batch are now totally flatten into (B * n, logits), so we have to divide the loss by batch_size

        return self.loss(logits.view(-1, logits.size(-1)), target.reshape(-1))


    def training_step(self, data: Tuple[torch.Tensor], batch_index:Any, *args: Any, **kwargs: Any) -> STEP_OUTPUT:

        loss = torch.zeros(1, device=self.device)
        if 'CLM' in data:
            clm_batch = data["CLM"]
            clm_input: torch.Tensor = clm_batch["input"]
            clm_target: torch.Tensor = clm_batch["label"]
            clm_attention_mask: torch.Tensor = clm_batch["attention_mask"]

            clm_loss = self._calculate_clm_loss(clm_input, clm_target, clm_attention_mask)
            loss += clm_loss
            

            
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # log train loss not too much frequently
        self.log("train_loss", loss)
        self.log("train_clm_loss", clm_loss)
        self.log("lr", lr)


        return {"batch_index": batch_index, "loss":loss}
    

    def validation_step(self, data: Tuple[torch.Tensor], batch_index,*args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        loss = torch.zeros(1, device=self.device)
        if 'CLM' in data:
            clm_batch = data["CLM"]
            clm_input: torch.Tensor = clm_batch["input"]
            clm_target: torch.Tensor = clm_batch["label"]
            clm_attention_mask: torch.Tensor = clm_batch["attention_mask"]

            clm_loss = self._calculate_clm_loss(clm_input, clm_target, clm_attention_mask)
            loss += clm_loss
            

        self.log("val_loss", loss)
        self.log("val_clm_loss", clm_loss)
        return {"batch_index": batch_index, "val_loss":loss}
    
    
    def test_step(self, data: Tuple[torch.Tensor], batch_index, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        loss = torch.zeros(1, device=self.device)
        if 'CLM' in data:
            clm_batch = data["CLM"]
            clm_input: torch.Tensor = clm_batch["input"]
            clm_target: torch.Tensor = clm_batch["label"]
            clm_attention_mask: torch.Tensor = clm_batch["attention_mask"]

            clm_loss = self._calculate_clm_loss(clm_input, clm_target, clm_attention_mask)
            loss += clm_loss
            

        self.log("test_loss", loss)
        self.log("test_clm_loss", clm_loss)
        return {"batch_index": batch_index, "val_loss":loss}
    
