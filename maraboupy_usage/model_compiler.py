import torch

def produce_py_file(n_tokens: int, num_heads=8, hidden_head_size=64):
class FixedAttentionMask(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn = model_og.gpt_neox.layers[0].attention
        self._init_bias(N_TOKENS)
        self.slice_query = Slicer(192, 0, 64)
        self.slice_value = Slicer(192, 64, 64)
        self.slice_key = Slicer(192, 64 * 2, 64)
        self.slice_rotary = Slicer(64, 0, 16)
        self.slice_non_rotary = Slicer(64, 16, 64 - 16)
        # TODO: I think 16 has to do with having 2 tokens... we need to generalize
        self.slice_rorate_half_1 = Slicer(16, 0, 8)
        self.slice_rorate_half_2 = Slicer(16, 8, 8)
        # QKV_last_size = (N_TOKENS,)
        # self.new_qkv_shape = (N_TOKENS, self.attn.num_attention_heads, 3 * self.attn.head_size)
        self.QKV_copier = QKVCopier() # TODO: dime sizew
        # ATTENTION BIAS ATTENTION BIAS 2 2 2 2 torch.Size([1, 1, 2048, 2048])
        # TODO: PARAMETERIZE BETTER
        # self.attn_bias_slice_a = Slicer(2048, 0, 2)
        # self.attn_bias_slice_b = Slicer(2048, 0, 2)
        # self.attn.bias = self.attn.bias[:, :, :2, :2]
    
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
            self._init_bias(N_TOKENS, device=key.device)
        # causal_mask = self.attn.bias[:, :, key_length - query_length : key_length, :key_length]
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
        position_ids=torch.arange(N_TOKENS)
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
            """Rotates half the hidden dims of the input."""
            print("ROTATE HALF SLICE DIMS", x.shape)
            # x1 = x[..., : x.shape[-1] // 2]
            # x2 = x[..., x.shape[-1] // 2 :]
            x1 = self.slice_rorate_half_1(x)
            x2 = self.slice_rorate_half_2(x)
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_embed(q, k, cos, sin, position_ids):
            cos = cos[position_ids]
            sin = sin[position_ids]
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed
 
        attn_outputs = []
        # TODO: we may have to unroll UHUHUH
        for i in range(self.attn.num_attention_heads):
            # _qkv = self.QKV_copier.slicers[i](qkv)
            lin_layer = self.QKV_copier._modules[f"slicer_{i}"]
            # _qkv = self.QKV_copier.slicer_0(qkv)
            _qkv = lin_layer(qkv)
            print("LOCAL", _qkv.shape)
            query = self.slice_query(_qkv)
            key = self.slice_key(_qkv)
            value = self.slice_value(_qkv)

            # Compute rotary embeddings on rotary_ndims
            # 16 and 64
            #query_rot = query[..., :self.attn.rotary_ndims]
            #query_pass = query[..., self.attn.rotary_ndims:]
            #key_rot = key[..., :self.attn.rotary_ndims]
            #key_pass = key[..., self.attn.rotary_ndims:]
            # print("LOCAL QKV", query.shape, key.shape, value.shape)
            query_rot = self.slice_rotary(query)
            query_pass = self.slice_non_rotary(query)
            key_rot = self.slice_rotary(key)
            key_pass = self.slice_non_rotary(key)

            # Compute token offset for rotary embeddings (when decoding)
            # print("SEQ LEN", key.shape)
            seq_len = key.shape[-2]
            cos, sin = self.attn.rotary_emb(value, seq_len=seq_len)
            print(query_rot.shape, key_rot.shape, cos.shape, sin.shape, position_ids.shape)
            query, key = apply_rotary_embed(query_rot, key_rot, cos, sin, position_ids)
            # TODO: idk if cat is OKAY
            query = torch.cat((query, query_pass), dim=-1)
            key = torch.cat((key, key_pass), dim=-1)

            # Cache QKV values
            # if has_layer_past:
            #     past_key = layer_past[0]
            #     past_value = layer_past[1]
            #     key = torch.cat((past_key, key), dim=-2)
            #     value = torch.cat((past_value, value), dim=-2)
            present = None
    
            # Compute attention
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, None)
            print("ATTN OUTPUT", attn_output.shape, query.shape, key.shape, value.shape)
            attn_outputs.append(attn_output)   
    
        # Reshape outputs
        # attn_output = self.attn._merge_heads(attn_output, self.attn.num_attention_heads, self.attn.head_size)
        attn_output = torch.cat(attn_outputs, dim=-1)
        print("ATTN OUTPUT", attn_output.shape, attn_outputs[0].shape)
        attn_output = self.attn.dense(attn_output)

        # outputs = (attn_output, present)
        # if output_attentions:
        #     outputs += (attn_weights,)

        return attn_output

        return self.attn(x, attention_mask=attention_mask, position_ids=torch.arange(N_TOKENS).unsqueeze(0))


embed_linear = torch.nn.Linear(
    embd_matrix.shape[0], embd_matrix.shape[1], bias=False)
embed_linear.weight = torch.nn.Parameter(embd_matrix.T)


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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_linear = embed_linear
        self.layer_norm = SimplfiedLayerNorm(model_og.gpt_neox.layers[0].input_layernorm)
        self.attn = FixedAttentionMask()

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


model_sel = ModelSel()
model_sel