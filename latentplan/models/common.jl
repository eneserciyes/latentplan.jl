export Chain, Linear, ReLU, GELU, Dropout, Embedding, one_hot, LayerNorm, softmax, mse_loss, MaxPool1d, paramlist, paramlist_decay, paramlist_no_decay

using Statistics: mean, var, std
using Knet.Ops21: gelu
using AutoGrad: @primitive

include("repeat_new.jl")
include("dropdims_new.jl")

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
paramlist(c::Chain) = Iterators.flatten(paramlist.(c.layers))
paramlist_decay(c::Chain) = Iterators.flatten(paramlist_decay.(c.layers))
paramlist_no_decay(c::Chain) = Iterators.flatten(paramlist_no_decay.(c.layers))

paramlist(c::Any) = []
paramlist_decay(c::Any) = []
paramlist_no_decay(c::Any) = []

mutable struct Linear; w; b; pdrop; bias; end
Linear(in_dim, out_dim; pdrop=0, init=(a...)->gaussian(a...;mean=0,std=0.02), bias=true) = Linear(param(out_dim,in_dim, init=init, atype=atype), param0(out_dim, atype=atype), pdrop, bias)
(l::Linear)(x) = reshape(l.w * reshape(dropout(x, l.pdrop), size(x)[1], :), size(l.w)[1], size(x)[2:end]...) .+ (l.bias ? l.b : 0)

paramlist(l::Linear) = Iterators.flatten([paramlist_decay(l), paramlist_no_decay(l)])
paramlist_decay(l::Linear) = [l.w]
paramlist_no_decay(l::Linear) = [l.b]

struct ReLU; end
(r::ReLU)(x) = relu.(x)

struct GELU; end
(g::GELU)(x) = gelu(x)

struct Dropout 
    pdrop
    function Dropout(pdrop=0.0)
        new(pdrop)
    end
end

(d::Dropout)(x) = dropout(x, d.pdrop)

mutable struct Embedding
    weight::Param{Matrix{Float32}}

    function Embedding(D, K)
        weight = Param(rand(Uniform(-1/K, 1/K), (D, K)))
        new(weight)
    end
end
paramlist(e::Embedding) = Iterators.flatten([paramlist_decay(e), paramlist_no_decay(e)])
paramlist_decay(e::Embedding) = []
paramlist_no_decay(e::Embedding) = [e.weight]

function (e::Embedding)(x)
    weight * transpose(x)
end

function one_hot(Type, indices, class_num)
    onehot = zeros(Type, class_num, size(indices)...)
    for index in CartesianIndices(indices)
        onehot[indices[index], index] = convert(Type, 1)
    end
    atype(onehot)
end

@primitive one_hot(Type, indices, class_num),dy,y nothing nothing nothing

mutable struct LayerNorm; a; b; ϵ; end
paramlist(l::LayerNorm) = [l.a, l.b]
paramlist_decay(l::LayerNorm) = []
paramlist_no_decay(l::LayerNorm) = [l.a, l.b]

function LayerNorm(dmodel; eps=Float32(1e-5))
    a = param(dmodel; init=ones, atype=atype)
    b = param(dmodel; init=zeros, atype=atype)
    LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = var(x,mean=μ,dims=1, corrected=false)
    # torch implementation takes sqrt after adding eps
    l.a .* (x .- μ) ./ sqrt.(σ .+ l.ϵ) .+ l.b                                                         
end

function softmax(w; dims)
    probs = exp.(w)
    return probs ./ sum(probs, dims=dims)
end

function mse_loss(x, y; reduction="mean")
    loss = (x .- y).^2
    if reduction == "mean"
        return mean(loss)
    elseif reduction == "sum"
        return sum(loss)
    elseif reduction == "none"
        return loss
    end
end

function mse_loss_detached(x, y; reduction="mean")
    return mse_loss(x, y; reduction=reduction)
end

@primitive mse_loss_detached(x, y; reduction="mean"),dy begin grad = dy .* 2.0f0 .* (x.-y); if reduction=="mean" grad ./ length(x) else grad end end nothing nothing

function binary_cross_entropy(probs,labels; reduction="mean")
    loss = -(labels .* log.(probs) .+ (1 .- labels) .* log.(1 .- probs))
    if reduction == "mean"
        mean(loss)
    elseif reduction == "sum"
        sum(loss)
    elseif reduction == "none"
        return loss
    end
end


struct MaxPool1d
    window;
    stride;
    
    function MaxPool1d(window, stride)
        new(window, stride)
    end
end

(m::MaxPool1d)(x) = begin
    pooled_size = Int(floor((size(x)[1] - m.window) / m.stride) + 1)
    pool_results = pool(reshape(x, size(x, 1), 1, 1, :); window=m.window, stride=m.stride)
    reshape(pool_results, pooled_size, size(x)[2:end]...)
end
