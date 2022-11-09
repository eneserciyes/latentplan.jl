module Transformers

include("common.jl")
using .Common: LayerNorm

struct SelfAttention
    key;
    query;
    value;
    attn_drop;
    resid_drop;
    proj;
    n_head;

    function SelfAttention(config)
    end
end

function(s::SelfAttention)(x)
end


struct AttentionBlock
    ln1;
    ln2;
    attn::SelfAttention;
    mlp;

    function AttentionBlock(config)
    end
end

function (a::AttentionBlock)(x)
end

struct CausalSelfAttention; 
    key; query; value; proj; mask;
    attn_drop; resid_drop;
    n_head;
    
    function CausalSelfAttention(config)
        key = Linear(config["n_embd"], config["n_embd"])
        query = Linear(config["n_embd"], config["n_embd"])
        value = Linear(config["n_embd"], config["n_embd"])
        proj = Linear(config["n_embd"], config["n_embd"])
        
        mask = Matrix(UpperTriangular(ones(config["block_size"],config["block_size"])))
        if haskey(config, "action_dim")
            joined_dim = config["observation_dim"] + config["action_dim"] + 2
            mask[joined_dim:joined_dim:end,:, :, :] .= 0
        end
        new(key,query,value,proj,mask, config["attn_pdrop"], config["resid_pdrop"], config["n_head"])
    end
end

function (c::CausalSelfAttention)(x)
    C, T, B = size(x)

    k = permutedims(reshape(c.key(x), (C ÷ c.n_head, c.n_head, T, B)), (1, 3, 2, 4)) # hs, T, nh, B
    q = permutedims(reshape(c.query(x), (C ÷ c.n_head, c.n_head, T, B)), (1, 3, 2, 4)) # hs, T, nh, B
    v = permutedims(reshape(c.value(x), (C ÷ c.n_head, c.n_head, T, B)), (1, 3, 2, 4)) # hs, T, nh, B
    
    # (T, hs, nh, B) x (hs, T, nh, B) -> (T, T, nh, B)
    att = bmm(permutedims(k, (2,1,3,4)), q) .* (1 / sqrt(size(k, 1)))
    att[c.mask[1:T,1:T] .== 0, :, :] .= -Inf
    att = softmax(att, dims=1)
    att_drop = dropout(att, c.attn_drop)
    # (hs, T, nh, B) x (T, T, nh, B)  -> (hs, T, nh, B)
    y = bmm(v, att_drop)
    # (C, T, B)
    y = reshape(permutedims(y, (1, 3, 2, 4)), (C, T, B)) # re-assemble all head outputs side by side
    # output projection
    y = dropout(c.proj(y), c.resid_drop)
    return y
end

struct Block
    ln1::LayerNorm;
    ln2::LayerNorm;
    attn::CausalSelfAttention;
    mlp::Chain;

    function Block(config)
        ln1 = LayerNorm(config["n_embd"])
        ln2 = LayerNorm(config["n_embd"])
        attn = CausalSelfAttention(config)
        mlp = Chain(
            Linear(config["n_embd"], 4 * config["n_embd"]), 
            GELU(),
            Linear(4 * config["n_embd"], config["n_embd"]),
            Dropout(config["resid_pdrop"])
        )
        new(ln1,ln2,attn,mlp)
    end
end

function (b::Block)(x)
    x = x .+  b.attn(b.ln1(x))
    x = x .+ b.mlp(b.ln2(x))
    return x
end

struct AsymBlock
    key;
    query;
    value;
    ln1;
    ln2;
    attention;
    mlp;

    function AsymBlock(config, out_tokens)
        # TODO: requires the implementation of nn.MultiheadAttention
    end
end

function (a::AsymBlock)(x)
    
end
