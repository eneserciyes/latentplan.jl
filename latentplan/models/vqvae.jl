module VQVAE

include("common.jl")

using .Common: Embedding, one_hot

# VectorQuantization
function vq(inputs::Array{Float32}, codebook::Matrix{Float32})
    embedding_size = size(codebook,1)
    inputs_size = size(inputs)
    inputs_flatten = reshape(inputs, (embedding_size, :))
    
    codebook_sqr = transpose(sum(codebook .^ 2, dims=1))
    inputs_sqr = sum(inputs_flatten .^ 2, dims=1)
    distances = (codebook_sqr .+ inputs_sqr) + -2 * (transpose(codebook) * inputs_flatten)
    indices_cartesian = argmin(distances, dims=1)
    indices_flatten = broadcast(x->x[1], indices_cartesian)
    indices = reshape(indices_flatten, inputs_size[2:end])
    return indices
end

# VectorQuantizationStraightThrough
function vq_st(inputs::Array{Float32}, codebook::Matrix{Float32})
    indices = vq(inputs, codebook)
    indices_flatten = reshape(indices, :)
    codes_flatten = codebook[:, indices]
    codes = reshape(codes_flatten, size(inputs))
    return codes, indices_flatten
end

# VectorQuantizationStraightThrough Backwards gradient calculation
function vq_st_codebook_backprop(codebook, output, grad_output)
    _, indices = output
    embedding_size = size(codebook, 1)
    grad_output_flatten = reshape(grad_output[1], (embedding_size, :))
    grad_codebook = zeros(Float32, size(codebook))
    grad_codebook[:, indices] += grad_output_flatten
    return grad_codebook
end

# gradient definition for straight through estimation
@primitive vq_st(inputs, codebook),dy,y dy[1] vq_st_codebook_backprop(codebook, y, dy)  


struct VQEmbedding
    embedding::Embedding

    function VQEmbedding(D, K)
        embedding = Embedding(D, K)
        new(embedding)
    end
end

function (v::VQEmbedding)(z_e_x::Array{Float32})
    latents = vq(z_e_x, v.embedding.weight)
    return latents
end

function (v::VQEmbedding)(z_e_x::Array{Float32}, straight_through::Bool)
    z_q_x, indices = vq_st(z_e_x, v.embedding.weight)
    z_q_x_bar = v.embedding.weight[:, indices]
    return z_q_x, z_q_x_bar
end

mutable struct VQEmbeddingMovingAverage
    embedding
    decay::Float32
    ema_count
    ema_w

    function VQEmbeddingMovingAverage(D, K; decay=0.99f0)
        embedding = rand(Uniform(-1/K, 1/K), (D, K))
        ema_count = ones(K)
        ema_w = deepcopy(embedding)
        new(embedding, decay, ema_count, ema_w)
    end
end

function (v::VQEmbeddingMovingAverage)(z_e_x)
    vq(z_e_x, v.embedding.weight)
end

function straight_through(v::VQEmbeddingMovingAverage, z_e_x, train::Bool=true)
    D, K = size(v.embedding)
    z_q_x, indices = vq_st(z_e_x, v.embedding)
    
    if train
        encodings = one_hot(Float32, indices, K)
        v.ema_count = v.decay .* v.ema_count + (1 - v.decay) .* sum(encodings, dims=2)[:, 1]
        dw = reshape(z_e_x, (D, :)) * transpose(encodings) 
        v.ema_w = v.decay .* v.ema_w + (1 - v.decay) .* dw
        v.embedding = v.ema_w ./ reshape(v.ema_count, (1, :))
        @size v.embedding
    end

    z_q_x_bar_flatten = v.embedding[:, indices]
    z_q_x_bar = reshape(z_q_x_bar_flatten, size(z_e_x))

    return z_q_x, z_q_x_bar
end




end
