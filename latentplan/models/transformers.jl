export CausalSelfAttention, Block, paramlist, paramlist_decay, paramlist_no_decay

using LinearAlgebra: UpperTriangular

struct CausalSelfAttention; 
    key::Linear; query::Linear; value::Linear; 
    proj::Linear; mask;
    attn_drop; resid_drop;
    n_head;
    
    function CausalSelfAttention(config)
        key = Linear(config["n_embd"], config["n_embd"])
        query = Linear(config["n_embd"], config["n_embd"])
        value = Linear(config["n_embd"], config["n_embd"])
        proj = Linear(config["n_embd"], config["n_embd"])
        
        mask = Matrix(UpperTriangular(ones(Float32, config["block_size"],config["block_size"])))
        if haskey(config, "action_dim")
            joined_dim = config["observation_dim"] + config["action_dim"] + 2
            mask[joined_dim:joined_dim:end,:, :, :] .= 0
        end
        new(key,query,value,proj,mask, config["attn_pdrop"], config["resid_pdrop"], config["n_head"])
    end
end
paramlist(c::CausalSelfAttention) = Iterators.flatten(paramlist.([c.key, c.query, c.value, c.proj]))
paramlist_decay(c::CausalSelfAttention) = Iterators.flatten(paramlist_decay.([c.key, c.query, c.value, c.proj]))
paramlist_no_decay(c::CausalSelfAttention) = Iterators.flatten(paramlist_no_decay.([c.key, c.query, c.value, c.proj]))

function (c::CausalSelfAttention)(x)
    C, T, B = size(x)

    k = permutedims(reshape(c.key(x), (C ÷ c.n_head, c.n_head, T, B)), (1, 3, 2, 4)) # hs, T, nh, B
    q = permutedims(reshape(c.query(x), (C ÷ c.n_head, c.n_head, T, B)), (1, 3, 2, 4)) # hs, T, nh, B
    v = permutedims(reshape(c.value(x), (C ÷ c.n_head, c.n_head, T, B)), (1, 3, 2, 4)) # hs, T, nh, B
    
    # (T, hs, nh, B) x (hs, T, nh, B) -> (T, T, nh, B)
    att = bmm(permutedims(k, (2,1,3,4)), q) .* Float32(1 / sqrt(size(k, 1)))
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
paramlist(b::Block) = Iterators.flatten(paramlist.([b.ln1, b.ln2, b.attn, b.mlp]))
paramlist_decay(b::Block) = Iterators.flatten(paramlist_decay.([b.ln1, b.ln2, b.attn, b.mlp]))
paramlist_no_decay(b::Block) = Iterators.flatten(paramlist_no_decay.([b.ln1, b.ln2, b.attn, b.mlp]))


function (b::Block)(x)
    x = x .+  b.attn(b.ln1(x))
    x = x .+ b.mlp(b.ln2(x))
    return x
end

#=
NOT TESTED
struct ScaledDotProductAttention; end

(s::ScaledDotProductAttention)(q,k,v) = begin
    dk = size(k, 1)
    scores = bmm(permutedims(k, (2,1,3)), q) .* (1 / sqrt(dk))
    att = softmax(scores, dims=1)
    return bmm(v, att)
end

# NOT TESTED
struct MultiHeadAttention
    embed_dim; num_head; linear_q; linear_k; linear_v; linear_o;
    
    function MultiHeadAttention(embed_dim, num_head)
        q = Linear(embed_dim, embed_dim)
        k = Linear(embed_dim, embed_dim)
        v = Linear(embed_dim, embed_dim)
        o = Linear(embed_dim, embed_dim)
        new(embed_dim, num_head, q,k,v,o)
    end 
end

function (m::MultiHeadAttention)(q,k,v)
    q = m.linear_q(q); k = m.linear_k(k); v = m.linear_v(v)
    q = _reshape_to_heads(m, q)
    k = _reshape_to_heads(m, k)
    v = _reshape_to_heads(m, v)
    y = ScaledDotProductAttention()(q,k,v)
    y = _reshape_from_heads(m, y)
    return m.linear_o(y)
end

function _reshape_to_heads(m::MultiHeadAttention, x)
    embed_dim, seq_len, batch_size = size(x)
    head_dim = embed_dim ÷ m.num_head
    x = reshape(x, head_dim, m.num_head, seq_len, batch_size)
    x = permutedims(x, (1, 3, 2, 4))
    return reshape(x, head_dim, seq_len, m.num_head * batch_size)
end

function _reshape_from_heads(m::MultiHeadAttention, x)
    head_dim, seq_len, batch_size = size(x)
    batch_size = batch_size / m.num_head
    out_dim = head_dim * m.num_head
    x = reshape(x, head_dim, seq_len, m.num_head, batch_size)
    x = permute(x, (1, 3, 2, 4))
    return reshape(x, out_dim, seq_len, batch_size)
end
    

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
=#
