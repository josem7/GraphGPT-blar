import torch as t
from torch import nn
import torch.nn.functional as F
import math
from transformers.configuration_utils import PretrainedConfig

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
class LayerNorm(nn.LayerNorm):
    """Subclass t's LayerNorm to handle fp16."""

    def forward(self, x: t.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(t.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: t.Tensor):
        return x * t.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: t.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: t.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: t.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: t.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: t.Tensor):
        return self.resblocks(x)
    
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = t.zeros(q_len, d_model)
    position = t.arange(0, q_len).unsqueeze(1)
    div_term = t.exp(t.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = t.sin(position * div_term)
    pe[:, 1::2] = t.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def pos_encoding(pe, learn_pe, nvar, d_model):
    # Positional encoding
    if pe == None:
        W_pos = t.empty((nvar, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = t.empty((nvar, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = t.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = t.zeros((nvar, 1))
        t.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = t.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(nvar, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class graph_transformer(nn.Module):
    def __init__(self, args):
        super(graph_transformer, self).__init__()
        self.config = PretrainedConfig()
        self.gtLayers = nn.Sequential(*[GTLayer(args) for i in range(args.gt_layers)])
        self.token_embedding = nn.Embedding(
            args.vocab_size, args.transformer_width
        )  # the embedding for all possible tokens
        self.positional_embedding = nn.Parameter(t.empty(self.context_length, args.transformer_width))
        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.text_projection = nn.Parameter(t.empty(args.transformer_width, args.embed_dim))
        self.W_pos = pos_encoding('zeros', True, 1, args.att_d_model)
                
        self.W_P = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.inverW_P = nn.Linear(args.att_d_model, args.gnn_output)
        self.args = args

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(
            1, 0, 2
        )  # NLD -> LND, batch_size * context_length *emb_dim -> context_length * batch_size  *emb_dim
        x = self.transformer(x)
        x = x.permute(
            1, 0, 2
        )  # LND -> NLD, context_length * batch_size *emb_dim -> batch_size * context_length *emb_dim
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot （end of token） embedding (eot_token is the highest number in each sequence)
        # so there is node need to shorten the context length
        x = x[t.arange(x.shape[0]), text.argmax(dim=-1)]  #
        x = x @ self.text_projection
        return x
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pyt uses additive attention mask; fill with -inf
        mask = t.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask
    
    def forward(self, g):
        # Adj: sp adj
        # x: bs * n * d_model * num_patch
        
        # print(edge_index)
        device = self.parameters().__next__().device
        g = g.to(device)
        
        x = g.x.bfloat16()
        name = x[-40:]
        x = x.encode_text(x)
        # x, W_P_weight, W_P_bias= Mv2Samedevice([x, self.W_P.weight, self.W_P.bias])
        # self.W_P.weight = nn.Parameter(W_P_weight.to(x.dtype))
        # self.W_P.bias = nn.Parameter(W_P_bias.to(x.dtype))
        z = self.W_P(x)
        if self.args.if_pos: 
            embeds = self.dropout(z + self.W_pos) 
        else: 
            embeds = self.dropout(z) 
        for gt in self.gtLayers:
            embeds = gt(g, embeds) # bs * num_patch * n * d_model
        # embeds, inverW_P_weight, inverW_P_bias = Mv2Samedevice([embeds, self.inverW_P.weight, self.inverW_P.bias])
        # self.inverW_P.weight = nn.Parameter(inverW_P_weight.to(embeds.dtype))
        # self.inverW_P.bias = nn.Parameter(inverW_P_bias.to(embeds.dtype))
        ret = self.inverW_P(embeds)
        return ret, name
    
def Mv2Samedevice(vars): 
    return [var.to(vars[0].device) for var in vars]

class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        if args.att_norm: 
            self.norm = nn.LayerNorm(args.att_d_model, eps=1e-6)
        self.args = args
        
        
    
    def forward(self, g, embeds):
        # Adj: adj
        # x: n * d_model
        rows, cols = g.edge_index
        nvar, _ = embeds.shape
        # print(rows)
        # print(cols)

        rowEmbeds = embeds[rows, :]
        colEmbeds = embeds[cols, :]
        evar, _ = rowEmbeds.shape

        # rowEmbeds, qTrans, kTrans, vTrans = Mv2Samedevice([rowEmbeds, self.qTrans, self.kTrans, self.vTrans])
        # self.qTrans = nn.Parameter(qTrans.to(rowEmbeds.dtype))
        # self.kTrans = nn.Parameter(kTrans.to(rowEmbeds.dtype))
        # self.vTrans = nn.Parameter(vTrans.to(rowEmbeds.dtype))
        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        
        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        
        tem = t.zeros([nvar, self.args.head]).to(expAtt.device, dtype=expAtt.dtype)
        # print(tem.device, expAtt.device, rows.device)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8) # bleh
        
        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([evar, self.args.att_d_model])
        tem = t.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device, dtype=resEmbeds.dtype)
        rows = rows.to(resEmbeds.device)
        tem = tem.to(resEmbeds.dtype)
        resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm: 
            # resEmbeds, norm_weight, norm_bias = Mv2Samedevice([resEmbeds, self.norm.weight, self.norm.bias])
            # self.norm.weight = nn.Parameter(norm_weight.to(resEmbeds.dtype))
            # self.norm.bias = nn.Parameter(norm_bias.to(resEmbeds.dtype))
            resEmbeds = self.norm(resEmbeds)

        return resEmbeds