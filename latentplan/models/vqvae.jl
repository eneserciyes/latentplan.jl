module VQVAE

include("common.jl")
include("transformers.jl")

using .Common: Embedding, one_hot, Chain, Linear, MaxPool1d, LayerNorm, Dropout, mse_loss
using .Transformers: Block, AsymBlock
using Statistics: mean
using Knet: Param, bce

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

#########################
# VQEmbedding

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

#########################
# VQStepWiseTransformer

struct VQStepWiseTransformer
    K
    latent_size
    condition_size
    trajectory_input_length
    embedding_dim
    trajectory_length
    block_size
    observation_dim
    action_dim
    transition_dim
    latent_step
    state_conditional
    masking
    encoder::Chain
    codebook::Union{VQEmbedding, VQEmbeddingMovingAverage}
    ma_update::Bool
    residual
    decoder::Chain
    pos_emb
    embed::Linear
    predict::Linear
    cast_embed::Linear
    latent_mixing::Linear
    bottleneck::String
    latent_pooling::Union{AsymBlock, MaxPool1d}
    expand
    ln_f::LayerNorm
    drop::Dropout

    function VQStepWiseTransformer(config, feature_dim)
        K = config["K"]
        latent_size = config["trajectory_embd"]
        condition_size = config["observation_dim"]
        trajectory_input_length = config["block_size"] - config["transition_dim"]
        embedding_dim = config["n_embd"]
        trajectory_length = config["block_size"] ÷ (config["transition_dim"]-1)
        block_size = config["block_size"]
        observation_dim = feature_dim
        action_dim = config["action_dim"]
        transition_dim = config["transition_dim"]
        latent_step = config["latent_step"]
        state_conditional = config["state_conditional"]
        if haskey(config, "masking")
            masking = config["masking"]
        else
            masking = "none"
        end

        encoder = Chain([Block(config) for _ in 1:config["n_layer"]]...)
        if haskey(config, "ma_update") && !(config["ma_update"])
            codebook = VQEmbedding(config["trajectory_embd"], config["K"])
            ma_update = false
        else
            codebook = VQEmbeddingMovingAverage(config["trajectory_embd"], config["K"])
            ma_update = true
        end

        if !haskey(config, "residual")
            residual = true
        else
            residual = config["residual"]
        end
        decoder = Chain([Block(config) for _ in 1:config["n_layer"]]...)
        pos_emb = Param(zeros(config["n_embd"], trajectory_length, 1))
        embed = Linear(transition_dim, embedding_dim)
        predict = Linear(embedding_dim, transition_dim)
        cast_embed = Linear(embedding_dim, transition_dim)
        latent_mixing = Linear(latent_size + observation_dim, embedding_dim)
        if !haskey(config, "bottleneck")
            bottleneck = "pooling"
        else
            bottleneck = config["bottleneck"]
        end
        
        if bottleneck == "pooling"
            latent_pooling = MaxPool1d(latent_step, latent_step)
            expand = nothing
        # else if bottleneck == "attention"
        #     latent_pooling = AsymBlock(config, trajectory_length ÷ latent_step)
        #     expand = AsymBlock(config, trajectory_length)
        end
        ln_f = LayerNorm(config["n_embd"])
        drop = Dropout(config["embd_pdrop"])

        new(
            K, latent_size, condition_size, trajectory_input_length, 
            embedding_dim, trajectory_length, block_size,
            observation_dim, action_dim, transition_dim, latent_step, 
            state_conditional, masking, encoder, codebook, 
            ma_update, residual, decoder, pos_emb, embed, predict, cast_embed, 
            latent_mixing, bottleneck, latent_pooling, expand, ln_f, drop
        )
    end
end


function encode(v::VQStepWiseTransformer, joined_inputs)
    joined_inputs = convert.(Float32, joined_inputs)
    _, t, _ = size(joined_inputs)
    @assert t <= v.block_size

    # forward the GPT model
    token_embeddings = v.embed(joined_inputs)

    ## [embedding_dim x T x 1]
    position_embeddings = v.pos_emb[:, 1:t, :]  # each position maps to a (learnable) vector
    ## [embedding_dim x T x B]
    x = v.drop(token_embeddings + position_embeddings)
    x = v.encoder(x)
    ## [embedding_dim x T x B]
    x = reshape(v.latent_pooling(permutedims(x, (2, 1, 3))), (2,1,3)) # pooling (not attention)
    ## [embedding_dim x (T//latent_step) x B]
    x = v.cast_embed(x)
    return x
end

function decode(v::VQStepWiseTransformer, latents, state)
    _, T, B = size(latents)
    state_flat = repeat(reshape(state, (:, 1, B)), 1, T, 1)
    if !v.state_conditional
        state_flat = zeros(size(state_flat))
    end
    inputs = cat((state_flat, latents), dims=1)
    inputs = v.latent_mixing(inputs)
    inputs = repeat(inputs, inner=(1, v.latent_step, 1))

    inputs = inputs + v.pos_emb[:, 1:size(inputs, 1), :]
    x = v.decoder(inputs)
    x = v.ln_f(x)

    ## [obs_dim x T x B]
    joined_pred = v.predict(x)
    joined_pred[end, :, :] = sigm.(joined_pred[end, :, :])
    joined_pred[1:v.observation_dim, :, :] += reshape(state, (B, 1, :))

    return joined_pred
end


function (v::VQStepWiseTransformer)(joined_inputs, state)
    trajectory_feature = encode(v,joined_inputs)
    latents_st, latents = straight_through(v.codebook, trajectory_feature)
    # no bottleneck attention here
    joined_pred = decode(v, latents_st, state)
    return joined_pred, latents, trajectory_feature
end


struct VQContinuousVAE
    model::VQStepWiseTransformer
    trajectory_embd
    vocab_size
    stop_token
    block_size
    observation_dim
    masking
    action_dim
    trajectory_length
    transition_dim
    action_weight
    reward_weight
    value_weight
    position_weight
    first_action_weight
    sum_reward_weight
    last_value_weight
    latent_step
    padding_vector

    function VQContinuousVAE(config)
        model = VQStepWiseTransformer(config, config["observation_dim"])
        trajectory_embd = config["trajectory_embd"]
        vocab_size = config["vocab_size"]
        stop_token = config["vocab_size"] * config["transition_dim"]
        block_size = config["block_size"]
        observation_dim = config["observation_dim"]
        if haskey(config, "masking")
            masking = config["masking"]
        else
            masking = "none"
        end
        action_dim = config["action_dim"]
        trajectory_length = config["block_size"] ÷ (config["transition_dim"]-1)
        transition_dim = config["transition_dim"]
        action_weight = config["action_weight"]
        reward_weight = config["reward_weight"]
        value_weight = config["value_weight"]
        position_weight = config["position_weight"]
        first_action_weight = config["first_action_weight"]
        sum_reward_weight = config["sum_reward_weight"]
        last_value_weight = config["last_value_weight"]
        latent_step = config["latent_step"]
        padding_vector = zeros(transition_dim - 1)
        new(
            model, trajectory_embd, vocab_size, stop_token, block_size, 
            observation_dim, masking, action_dim, trajectory_length, 
            transition_dim, action_weight, reward_weight, value_weight, 
            position_weight, first_action_weight, sum_reward_weight, 
            last_value_weight, latent_step, padding_vector
        )
    end
end

function encode(v::VQContinuousVAE, joined_inputs, terminals)
    _, t, b = size(joined_inputs)
    padded = repeat(v.padding_vector, (1, t, b))
    terminal_mask = repeat(deepcopy(1 .- terminals), (size(joined_inputs, 1), 1, 1))
    joined_inputs = joined_inputs .* terminal_mask .+ padded .* (1 .- terminal_mask)

    trajectory_feature = encode(v.model, cat((joined_inputs, terminals), dims=1)) # TODO: check dims here
    if v.model.ma_update
        indices = vq(trajectory_feature, v.model.codebook.embedding)
    else
        indices = vq(trajectory_feature, v.model.codebook.embedding.weight)
    end
    return indices
end

function decode(v::VQContinuousVAE, latent, state)
    return decode(v.model, latent, state)
end

## TODO: decode_from_indices if necessary


function (v::VQContinuousVAE)(joined_inputs; targets=nothing, mask=nothing, terminals=nothing)
    joined_inputs = convert(Float32, joined_inputs)
    joined_dimension, t, b = size(joined_inputs)
    padded = repeat(convert(Float32, v.padding_vector), (1, t, b))

    if !(terminals === nothing)
        terminal_mask = repeat(deepcopy(1 .- terminals), (size(joined_inputs, 1), 1, 1))
        joined_inputs = joined_inputs .* terminal_mask .+ padded .* (1 .- terminal_mask)
    end
    state = joined_inputs[1:v.observation_dim, 1, :]
    ## [ embedding_dim X T x B ]

    # forward the GPT model
    reconstructed, latents, feature = v.model(cat((joined_inputs, terminals), dims=1), state)
    pred_trajectory = reshape(reconstructed[1:end-1, :, :], (joined_dimension, t, b))
    pred_terminals = reshape(reconstructed[end, :, :], 1,1,size(reconstructed)[2:end]...)

    if !(targets === nothing)
        # compute the loss
        weights = cat([
            ones(2) .* v.position_weight,
            ones(v.observation_dim - 2),
            ones(v.action_dim) .* v.action_weight,
            ones(1) .* v.reward_weight,
            ones(1) .* v.value_weight,
        ])

        mse = mse_loss(pred_trajectory, joined_inputs, reduction="none") .* reshape(weights, (:, 1, 1))
        first_action_loss = v.first_action_weight .* mse_loss(
            joined_inputs[observation_dim:observation_dim+action_dim, 1, :], 
            pred_trajectory[observation_dim:observation_dim+action_dim, 1, :]
        )
        sum_reward_loss = v.sum_reward_weight .* mse_loss(
            mean(joined_inputs[end-1, :, :], dims=1), 
            mean(pred_trajectory[end-1, :, :], dims=1)
        )
        last_value_loss = v.last_value_weight .* mse_loss(
            joined_inputs[end, end, :], 
            pred_trajectory[end, end, :]
        )
        cross_entropy = bce(pred_terminals, clamp.(convert(Float32, terminals),0.0, 1.0))
        reconstruction_loss = mean((mse .* mask .* terminal_mask)) + cross_entropy
        reconstruction_loss = reconstruction_loss + first_action_loss + sum_reward_loss + last_value_loss

        if v.model.ma_update
            loss_vq = 0
        else
            loss_vq = mse_loss(latents, feature)
        end
        loss_commit = mse_loss(feature, latents)
    else
        reconstruction_loss = nothing
        loss_vq = nothing
        loss_commit = nothing
        return reconstructed, reconstruction_loss, loss_vq, loss_commit
    end
end

end
