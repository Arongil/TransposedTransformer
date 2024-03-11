"""
Transposed Transformer -- an invention of Professor Phillip Isola (MIT)

Normally transformers share weights across the context window.
What if we shared weights across depth instead? What could we learn?

The transposed transformer can stack more layers than it trained with,
just as normal transformers can read more context than they trained with.

It's not autoregressive, but instead passes information all around.
It's something like a mixture of experts, governed by attention.

What are the scaling laws? What are the dynamics?
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)

class PositionDependentLinear(nn.Module):
    """ Linear but with different parameters for every token position. """
    
    def __init__(self, n_tokens, n_in, n_out):
        super().__init__()
        stdev = 1/(n_out**0.5)
        self.W = nn.Parameter(stdev * torch.randn(n_tokens, n_in, n_out))
        self.b = nn.Parameter(torch.zeros(n_tokens, n_out))

    def forward(self, x):
        # Input shape  (batch, token_pos, n_in)
        # Output shape (batch, token_pos, n_out), where a different linear layer acts on each token_pos
        return torch.einsum('bti,tio->bto', x, self.W) + self.b

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = PositionDependentLinear(config.n_tokens, config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = PositionDependentLinear(config.n_tokens, config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = config.is_causal
        self.register_buffer("bias", torch.tril(torch.ones(config.n_tokens, config.n_tokens))
                                    .view(1, 1, config.n_tokens, config.n_tokens))

    def forward(self, x):
        return x
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch, then move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=-1) # (T, C, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(2, 3) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(2, 3) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(2, 3) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T), attention scores for all token pairs
        if self.is_causal:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # mask autoregressively only if is_causal is True
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.f_in = PositionDependentLinear(config.n_tokens, config.n_embd, 4 * config.n_embd)
        self.f_out = PositionDependentLinear(config.n_tokens, 4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return x
        x = self.f_in(x)
        x = self.gelu(x)
        x = self.f_out(x)
        x = self.dropout(x)
        return x

### TT (Transposed Transformer) ###

@dataclass
class TTConfig:
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_tokens: int = 8       # fixed context window -- cannot be modified due to weight sharing across depth!
    n_layer: int = 4        # number of layers is variable due to weight sharing across depth!
    n_head: int = 8
    n_embd: int = 64*3
    dropout: float = 0.0
    is_causal: bool = False # more natural to let all token positions communicate, so not autoregressive

class TT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_tokens is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            drop = nn.Dropout(config.dropout),
            attention = SelfAttention(config),
            mlp = MLP(config),
            ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=False),
            ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=False),
            ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t == self.config.n_tokens, f"Cannot forward sequence of length {t}, only of length n_tokens = {self.config.n_tokens}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)  # no positional embeddings since every token position is already distinguished
        for _ in range(self.config.n_layer):
            x = x + self.transformer.attention(self.transformer.ln_1(x))
            x = x + self.transformer.mlp(self.transformer.ln_2(x))
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, config):
        weight_decay, learning_rate, betas = config.weight_decay, config.learning_rate, config.betas
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        #fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        #use_fused = fused_available and device_type == 'cuda'
        #extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)#, **extra_args)
        #print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.n_tokens
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at n_tokens
            idx_cond = idx if idx.size(1) <= self.config.n_tokens else idx[:, -self.config.n_tokens:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

