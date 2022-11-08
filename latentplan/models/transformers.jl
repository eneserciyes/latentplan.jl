module Transformers

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

struct CausalSelfAttention
    key;
    query;
    value;
    attn_drop;
    resid_drop;
    proj;
    mask;
    n_head;

    function CausalSelfAttention(config)
    end 
end

function (c::CausalSelfAttention)(x)
end

struct Block
    ln1;
    ln2;
    attn::CausalSelfAttention;
    mlp;

    function Block(config)
    end
end

function (b::Block)(x)
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
