module Common

using Knet
using Distributions: Normal, cdf, mean, std

export Chain
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

export Linear
struct Linear; w; b; pdrop; end
Linear(in_dim::Int, out_dim::Int; pdrop=0) = Linear(param(out_dim,in_dim), param0(out_dim), pdrop)
(l::Linear)(x) = reshape(l.w * reshape(dropout(x, l.pdrop), size(x)[1], :), size(l.w)[1], size(x)[2:end]...) .+ l.b

paramlist(l::Linear) = Iterators.flatten([paramlist_decay(l), paramlist_no_decay(l)])
paramlist_decay(l::Linear) = [l.w]
paramlist_no_decay(l::Linear) = [l.b]

export ReLU
struct ReLU; end
(r::ReLU)(x) = relu.(x)

export GELU
struct GELU; end
(g::GELU)(x) = x .* cdf.(Normal(), x)

struct Dropout 
    pdrop
    function Dropout(pdrop=0.0)
        new(pdrop)
    end
end

(d::Dropout)(x) = dropout(x, d.pdrop)


export Embedding
struct Embedding
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

export one_hot
function one_hot(Type, indices, class_num)
    onehot = zeros(Type, class_num, size(indices)...)
    for index in CartesianIndices(indices)
        onehot[indices[index], index] = convert(Type, 1)
    end
    onehot
end

export LayerNorm
struct LayerNorm; a; b; ϵ; end
paramlist(l::LayerNorm) = Dict("no_decay" => [l.a, l.b], "decay" => [])
paramlist_decay(l::LayerNorm) = []
paramlist_no_decay(l::LayerNorm) = [l.a, l.b]

function LayerNorm(dmodel; eps=1e-5)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = std(x,mean=μ,dims=1, corrected=false)
    l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b                                                         
end

export softmax
function softmax(w; dims::Int)
    probs = exp.(w)
    return probs ./ sum(probs, dims=dims)
end

export mse_loss
function mse_loss(x, y; reduction="mean")
    if reduction == "mean"
        return sum((x .- y).^2) / size(x)[end]
    elseif reduction == "sum"
        return sum((x .- y).^2)
    elseif reduction == "none"
        return sum((x .- y).^2, dims=1:ndims(x)-1)
    end
end


# TODO: wrong implementation, check this again
export MaxPool1d
struct MaxPool1d
    window;
    stride;
    
    function MaxPool1d(window, stride)
        new(window, stride)
    end
end

(m::MaxPool1d)(x) = begin 
    pool_results = pool(reshape(x, size(x, 1), 1, 1); window=m.window, stride=m.stride)[:,1,1]
    reshape(pool_results, size(pool_results, 1), size(x)[2:end]...) 
end

end



