module Common

using Knet

export Chain
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

export Linear
struct Linear; w; b; pdrop; end
Linear(in_dim::Int, out_dim::Int; pdrop=0) = Linear(param(out_dim,in_dim), param0(out_dim), pdrop)
(l::Linear)(x) = reshape(l.w * reshape(dropout(x, l.pdrop), size(x)[1], :), size(l.w)[1], size(x)[2:end]...) .+ l.b
paramlist(l::Linear) = [l.w, l.b]


export ReLU
struct ReLU; end
(r::ReLU)(x) = relu.(x)

export Embedding
struct Embedding
    weight::Param{Matrix{Float32}}

    function Embedding(D, K)
        weight = Param(rand(Uniform(-1/K, 1/K), (D, K)))
        new(weight)
    end
end

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

end



