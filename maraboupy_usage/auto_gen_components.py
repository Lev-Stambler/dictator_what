IMPORT_START = """
import torch
from components import Slicer
"""


def gen_qkv_copier(save_path: str, n_tokens: int, num_heads: int, hidden_size: int):
    start_str = IMPORT_START + f"""
class QKVCopier(torch.nn.Module):
    def __init__(self, seq_len={n_tokens}, num_heads={num_heads}, hidden_size={hidden_size}) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_size = hidden_size"""
    for i in range(num_heads):
        start_str += f"""
        i = {i}
        self.slicer_{i}  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)"""
    open(save_path, 'w').write(start_str + '\n')
    return start_str


def gen_rotary_embedding(save_path: str, seq_len: int):
    start_str = IMPORT_START + f"""
class GPTNeoXRotaryEmbeddingModified(torch.nn.Module):
    # Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding.__init__
    def __init__(self, dim, max_position_embeddings={seq_len}, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq = torch.nn.Parameter(1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)))

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len={seq_len}, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = torch.nn.Parameter(emb.cos())
        self.sin_cached = torch.nn.Parameter(emb.sin())
        # self.register_buffer("cos_cached", emb.cos(), persistent=False)
        # self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x):
        # x: [bs, num_attention_heads, {seq_len}, head_size]
        # We use a fixed seq len here
        # if seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # self.cos_cached[:{seq_len}],
        # self.sin_cached[:{seq_len}],
        return (
            self.cos_cached,
            self.sin_cached,
        )

"""
    open(save_path, 'w').write(start_str + '\n')


def gen_attn_head_unrolled(save_path: str, qkv_copier_path: str, rotary_embed_path: str, n_tokens: int, num_heads: int, head_hidden_size: int):
    start_str = IMPORT_START + f"""
from {qkv_copier_path} import QKVCopier
from {rotary_embed_path} import GPTNeoXRotaryEmbeddingModified
""" + f"""
class FixedAttentionMask(torch.nn.Module):

    def __init__(self, attention_head_og) -> None:
        super().__init__()
        # TODO: IDK HERE!!! 16 always?
        self.rotary_dim = 16
        self.attn = attention_head_og
        self._init_bias({n_tokens})
        self.slice_query = Slicer({3 * head_hidden_size}, 0, {head_hidden_size})
        self.slice_value = Slicer({3 * head_hidden_size}, {head_hidden_size}, {head_hidden_size})
        self.slice_key = Slicer({3 * head_hidden_size}, {head_hidden_size} * 2, {head_hidden_size})
        self.slice_rotary = Slicer({head_hidden_size}, 0, self.rotary_dim)
        self.slice_non_rotary = Slicer({head_hidden_size}, self.rotary_dim, {head_hidden_size} - self.rotary_dim)
        self.slice_rorate_half_1 = Slicer(self.rotary_dim, 0, self.rotary_dim // 2)
        self.slice_rorate_half_2 = Slicer(self.rotary_dim, self.rotary_dim // 2, self.rotary_dim // 2)
        # QKV_last_size = ({n_tokens},)
        # self.new_qkv_shape = ({n_tokens}, self.attn.num_attention_heads, 3 * self.attn.head_size)
        self.QKV_copier = QKVCopier()
        self.rotary_emb = GPTNeoXRotaryEmbeddingModified(self.rotary_dim)
        self.attention_mask = torch.ones((1, {n_tokens}), dtype=torch.int)
        self.flip_sign = torch.nn.Parameter(torch.tensor(-1.0))
    
    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)



    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # TODO: MAKE THIS NON BATCH BASED
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias({n_tokens}, device=key.device)
        causal_mask = self.bias
        # A cheap way of doing the above: TODO: MAYBE HARDCODE THIS IN!

        # query = query.reshape(num_attention_heads, query_length, attn_head_size)
        # key = key.reshape(num_attention_heads, key_length, attn_head_size)
        # query = query.view(num_attention_heads, query_length, attn_head_size)
        # key = key.view(num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        # print("ATTENTION BIAS", key_length, query_length, key_length, key_length, self.bias.shape)
        attn_scores = attn_scores + (query @ key.transpose(0, 1)) * self.attn.norm_factor
        # attn_scores = torch.baddbmm(
        #     attn_scores,
        #     query,
        #     key.transpose(0, 1),
        #     beta=1.0,
        #     alpha=self.attn.norm_factor,
        # )
        # print("ATTTN SCORES", attn_scores.shape, attn_scores)
        attn_scores = attn_scores.reshape(query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # todo: put back in
        # attn_weights = self.attn.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights



    def forward(self, x):
        # TODO: IDK IF THIS IS RIGHT for POSITION IDS or ATTENTION MASK
        position_ids=torch.arange({n_tokens})
        # Compute QKV
        # Attention heads [seq_len, hidden_size] --> [seq_len, (np * 3 * head_size)]
        qkv = self.attn.query_key_value(x)
        print(qkv.shape)

        # [seq_len, (num_heads * 3 * head_size)] --> [num_heads, num_tokens, 3 * head_size]
        print("QKV SIZE", qkv.size(), qkv.size()[:-1])
        # TODO: I think that this has to be a parameter
        # qkv = qkv.reshape(*self.new_qkv_shape)
        # List of qkv tensors
        # qkv = self.QKV_copier(qkv)
        # [seq_len, num_attention_heads, 3 * head_size] --> 3 [num_attention_heads, seq_len, head_size]
        # query = qkv[..., :self.attn.head_size].permute(1, 0, 2)

        def rotate_half(x):
            # Rotates half the hidden dims of the input.
            x1 = self.slice_rorate_half_1(x)
            x2 = self.slice_rorate_half_2(x) * self.flip_sign
            return torch.cat((x2, x1), dim=-1)

        def apply_rotary_embed(q, k, cos, sin, position_ids):
            cos = cos[position_ids]
            sin = sin[position_ids]
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed
 
        """
    for i in range(num_heads):
        start_str += f"""
        attns_output_{i} = None
        lin_layer = self.QKV_copier.slicer_{i}
        _qkv = lin_layer(qkv)
        query = self.slice_query(_qkv)
        key = self.slice_key(_qkv)
        value = self.slice_value(_qkv)

        # Compute rotary embeddings on rotary_ndims
        query_rot = self.slice_rotary(query)
        query_pass = self.slice_non_rotary(query)
        key_rot = self.slice_rotary(key)
        key_pass = self.slice_non_rotary(key)

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        # TODO: IS THIS PROBLEM?
        cos, sin = self.rotary_emb(value)
        print(query_rot.shape, key_rot.shape, cos.shape, sin.shape, position_ids.shape)
        query, key = apply_rotary_embed(query_rot, key_rot, cos, sin, position_ids)

        # TODO: idk if cat is OKAY
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)
        # Cache QKV values
        # TODO:
        # if has_layer_past:
        #     past_key = layer_past[0]
        #     past_value = layer_past[1]
        #     key = torch.cat((past_key, key), dim=-2)
        #     value = torch.cat((past_value, value), dim=-2)
        present = None
        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, self.attention_mask, None)
        print("ATTN OUTPUT", attn_output.shape, query.shape, key.shape, value.shape)
        attn_outputs_{i} = attn_output
        """

    start_str += f"""
        # Reshape outputs
        attn_output = torch.cat(({", ".join("attn_outputs_" + str(i) for i in range(num_heads))}), dim=-1)
        attn_output = self.attn.dense(attn_output)

        # outputs = (attn_output, present)
        # if output_attentions:
        #     outputs += (attn_weights,)

        return attn_output
"""
    # Save the string
    open(save_path, 'w').write(start_str + '\n')


if __name__ == '__main__':
    QKV_path = 'tmp/qkv_copier.py'
    rotary_path = 'tmp/rotary_embedding.py'
    gen_qkv_copier(QKV_path, n_tokens=2, num_heads=8, hidden_size=64)
    gen_rotary_embedding(rotary_path, seq_len=2)
    gen_attn_head_unrolled('tmp/attn_head.py', 'tmp.qkv_copier',
                           'tmp.rotary_embedding',
                           n_tokens=2, num_heads=8, head_hidden_size=64)
