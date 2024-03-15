import torch
from tmp.attn_head import FixedAttentionMask

class SimplfiedLayerNorm(torch.nn.Module):
    def __init__(self, layernorm: torch.nn.LayerNorm) -> None:
        super().__init__()
        self.weight = layernorm.weight
        self.bias = layernorm.bias
        self.eps = torch.nn.Linear(512, 512)
        self.eps.weight = torch.nn.Parameter(torch.eye(512))
        self.eps.bias = torch.nn.Parameter(torch.ones(512) * 1e-5)
        # TODO: not constant
        self.ones_linear = torch.nn.Linear(512, 512, bias=False)
        self.ones_linear.weight = torch.nn.Parameter(torch.ones((512, 512)))
        self.ones_linear_neg = torch.nn.Linear(512, 512, bias=False)
        self.ones_linear_neg.weight = torch.nn.Parameter(-1 * torch.ones((512, 512)))

    def forward(self, x):
        # TODO: I think that this can be made more efficient
        expectation_neg = self.ones_linear_neg(x)
        variance = self.ones_linear((((x + expectation_neg) * (x + expectation_neg))))
        radical = self.eps(variance)
        denom = torch.sqrt(radical)
        x = x + expectation_neg
        # TODO: DENOM HAS PROVLEMS
        return x
        x = x / denom
        return x
        x = x * self.weight
        x = x + self.bias

        return x


class ModelSel(torch.nn.Module):
    def __init__(self, model_og: torch.nn.Module) -> None:
        super().__init__()
        embd_matrix = model_og.gpt_neox.embed_in.weight
        embed_linear = torch.nn.Linear(
            embd_matrix.shape[0], embd_matrix.shape[1], bias=False)
        embed_linear.weight = torch.nn.Parameter(embd_matrix.T)
        self.embed_linear = embed_linear
        self.layer_norm = SimplfiedLayerNorm(model_og.gpt_neox.layers[0].input_layernorm)
        self.attn = FixedAttentionMask(model_og.gpt_neox.layers[0].attention)

    def forward(self, x):
        x = self.embed_linear(x)
        x = self.layer_norm(x)
        # HRMM Unsqueeze no good
        # x = x.reshape((1, *x.shape))
        x = self.attn(x)
        return x

# G
# # TODO: add residuals?
# model_sel = torch.nn.Sequential(
#     # model_og.gpt_neox.embed_in,
#     embed_linear,
#     # model_og.gpt_neox.emb_dropout, # we have p = 0.0 and thus useless
#     model_og.gpt_neox.layers[0].input_layernorm,
#     FixedAttentionMask(),
#     # TODO: VERIFY THIS JAZZ
# )

