export VQEmbeddingMovingAverage, VQEmbedding, VQStepWiseTransformer, VQContinuousVAE, paramlist, paramlist_no_decay, paramlist_decay

using Statistics: mean
using AutoGrad
using Distributions: Uniform
using Debugger: @bp

# VectorQuantization
function vq(inputs::atype, codebook::atype)
    embedding_size = size(codebook,1)
    inputs_size = size(inputs)
    inputs_flatten = reshape(inputs, (embedding_size, :))
    
    codebook_sqr = dropdims(sum(codebook .^ 2, dims=1), dims=1)
    inputs_sqr = sum(inputs_flatten .^ 2, dims=1)
    distances = (codebook_sqr .+ inputs_sqr) + -2 * (transpose(codebook) * inputs_flatten)
    indices_cartesian = argmin(distances, dims=1)
    indices_flatten = broadcast(x->x[1], indices_cartesian)
    indices = reshape(indices_flatten, inputs_size[2:end])
    return indices
end
@zerograd vq(inputs::atype, codebook::atype)

# VectorQuantizationStraightThrough
function vq_st(inputs::atype, codebook::atype)
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
    grad_codebook = atype(zeros(Float32, size(codebook)))
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

paramlist(v::VQEmbedding) = paramlist(v.embedding)
paramlist_decay(v::VQEmbedding) = paramlist_decay(v.embedding)
paramlist_no_decay(v::VQEmbedding) = paramlist_no_decay(v.embedding)

function (v::VQEmbedding)(z_e_x::atype)
    latents = vq(z_e_x, v.embedding.weight)
    return latents
end

function (v::VQEmbedding)(z_e_x::atype, straight_through::Bool)
    z_q_x, indices = vq_st(z_e_x, v.embedding.weight)
    z_q_x_bar = v.embedding.weight[:, indices]
    return z_q_x, z_q_x_bar
end

mutable struct VQEmbeddingMovingAverage
    embedding
    decay
    ema_count
    ema_w

    function VQEmbeddingMovingAverage(D, K; decay=0.99f0)
        embedding = atype(Float32.(rand(Uniform(-1/K, 1/K), (D, K))))
        ema_count = atype(ones(Float32, K))
        ema_w = deepcopy(embedding)
        new(embedding, decay, ema_count, ema_w)
    end
end
paramlist(v::VQEmbeddingMovingAverage) = []
paramlist_decay(v::VQEmbeddingMovingAverage) = []
paramlist_no_decay(v::VQEmbeddingMovingAverage) = []

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
    end

    z_q_x_bar_flatten = v.embedding[:, indices]
    z_q_x_bar = reshape(z_q_x_bar_flatten, size(z_e_x))

    return z_q_x, z_q_x_bar
end

#########################
# VQStepWiseTransformer

mutable struct VQStepWiseTransformer
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
    state_conditional::Bool
    masking::String
    encoder::Chain
    codebook::Union{VQEmbedding, VQEmbeddingMovingAverage}
    ma_update::Bool
    residual::Bool
    decoder::Chain
    pos_emb::Param
    embed::Linear
    predict::Linear
    cast_embed::Linear
    latent_mixing::Linear
    bottleneck::String
    latent_pooling::MaxPool1d
    expand
    ln_f::LayerNorm
    drop::Dropout

    function VQStepWiseTransformer(config, feature_dim)
        K = config["K"]
        latent_size = config["trajectory_embd"]
        condition_size = config["observation_dim"]
        trajectory_input_length = config["block_size"] - config["transition_dim"]
        embedding_dim = config["n_embd"]
        trajectory_length = config["block_size"] ÷ config["transition_dim"] - 1
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
        pos_emb = Param(atype(zeros(Float32, config["n_embd"], trajectory_length, 1)))
        embed = Linear(transition_dim, embedding_dim)
        predict = Linear(embedding_dim, transition_dim)
        cast_embed = Linear(embedding_dim, latent_size)
        latent_mixing = Linear(latent_size + observation_dim, embedding_dim)
        if !haskey(config, "bottleneck")
            bottleneck = "pooling"
        else
            bottleneck = config["bottleneck"]
        end
        
        if bottleneck == "pooling"
            latent_pooling = MaxPool1d(latent_step, latent_step)
            expand = nothing
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

paramlist(v::VQStepWiseTransformer) = begin
    model_params = collect(Param,
        Iterators.flatten(
            paramlist.([v.encoder, v.decoder, v.codebook, v.embed, v.predict,v.cast_embed, v.latent_mixing, v.ln_f])
        )
    )
    push!(model_params, v.pos_emb)
    model_params
end
paramlist_no_decay(v::VQStepWiseTransformer) = begin
    model_params = collect(Param,
        Iterators.flatten(
            paramlist_no_decay.([v.encoder, v.decoder, v.codebook, v.embed, v.predict,v.cast_embed, v.latent_mixing, v.ln_f])
        )
    )
    push!(model_params, v.pos_emb)
    model_params
end
paramlist_decay(v::VQStepWiseTransformer) = Iterators.flatten(
    paramlist_decay.([v.encoder, v.decoder, v.codebook, v.embed, v.predict,v.cast_embed, v.latent_mixing, v.ln_f]), 
)

function encode(v::VQStepWiseTransformer, joined_inputs)
    _, t, _ = size(joined_inputs)
    @assert t <= v.block_size

    # forward the GPT model
    token_embeddings = v.embed(joined_inputs)

    ## [embedding_dim x T x 1]
    position_embeddings = v.pos_emb[:, 1:t, :]  # each position maps to a (learnable) vector
    ## [embedding_dim x T x B]
    x = v.drop(token_embeddings .+ position_embeddings)
    x = v.encoder(x)
    ## [embedding_dim x T x B]
    x = permutedims(v.latent_pooling(permutedims(x, (2, 1, 3))), (2,1,3)) # pooling (not attention)
    ## [embedding_dim x (T//latent_step) x B]
    x = v.cast_embed(x)
    return x
end

function decode(v::VQStepWiseTransformer, latents, state)
    _, T, B = size(latents)
    state_flat = repeat_broadcast(reshape(state, (:, 1, B)), 1, T, 1)
    if !v.state_conditional
        state_flat = zeros(Float32, size(state_flat))
    end
    inputs = cat(state_flat, latents, dims=1)
    inputs = v.latent_mixing(inputs)
    inputs = repeat_interleave(inputs, 1, v.latent_step, 1)

    inputs = inputs .+ v.pos_emb[:, 1:size(inputs, 2), :]

    x = v.decoder(inputs)
    x = v.ln_f(x)

    ## [obs_dim x T x B]
    joined_pred = v.predict(x)
    
    sigm_mask = atype(ones(size(joined_pred)))
    sigm_mask[end, :, :] .= 0.0f0
    joined_pred = joined_pred .* sigm_mask .+ (1 .- sigm_mask) .* sigm.(joined_pred)
    
    state_mask = atype(zeros(size(joined_pred)))
    state_mask[1:v.observation_dim, :, :] .+= reshape(state, (:, 1, B))
    joined_pred += state_mask
    
    return joined_pred
end


function (v::VQStepWiseTransformer)(joined_inputs, state)
    trajectory_feature = encode(v, joined_inputs)
    latents_st, latents = straight_through(v.codebook, trajectory_feature)
    # # no bottleneck attention here
    joined_pred = decode(v, latents_st, state)
    return joined_pred, latents, trajectory_feature
end


mutable struct VQContinuousVAE
    model::VQStepWiseTransformer
    trajectory_embd
    vocab_size
    stop_token
    block_size
    observation_dim
    masking::String
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
    padding_vector::Vector

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
        trajectory_length = config["block_size"] ÷ config["transition_dim"] - 1
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

paramlist(v::VQContinuousVAE) = paramlist(v.model)
paramlist_decay(v::VQContinuousVAE) = paramlist_decay(v.model)
paramlist_no_decay(v::VQContinuousVAE) = paramlist_no_decay(v.model)

function encode(v::VQContinuousVAE, joined_inputs, terminals)
    _, t, b = size(joined_inputs)
    padded = repeat_broadcast(v.padding_vector, 1, t, b)
    terminal_mask = repeat_broadcast(deepcopy(1 .- terminals), size(joined_inputs, 1), 1, 1)
    joined_inputs = joined_inputs .* terminal_mask .+ padded .* (1 .- terminal_mask)

    trajectory_feature = encode(v.model, cat((joined_inputs, terminals), dims=1))
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


function (v::VQContinuousVAE)(joined_inputs, targets=nothing, mask=nothing, terminals=nothing)
    joined_dimension, t, b = size(joined_inputs)
    padded = repeat_broadcast(v.padding_vector, 1, t, b)

    if !(terminals === nothing)
        terminal_mask = repeat_broadcast(deepcopy(1 .- terminals), size(joined_inputs, 1), 1, 1)
        joined_inputs = joined_inputs .* terminal_mask .+ padded .* (1 .- terminal_mask)
    end
    state = joined_inputs[1:v.observation_dim, 1, :]
    ## [ embedding_dim X T x B ]

    # forward the GPT model
    reconstructed, latents, feature = v.model(cat(joined_inputs, terminals, dims=1), state)

    pred_trajectory = reshape(reconstructed[1:end-1, :, :], (joined_dimension, t, b))
    pred_terminals = reshape(reconstructed[end, :, :], 1,size(reconstructed)[2:end]...)
    if !(targets === nothing)
        # compute the loss
        weights = cat(
            atype(ones(2)) .* v.position_weight,
            atype(ones(v.observation_dim - 2)),
            atype(ones(v.action_dim)) .* v.action_weight,
            atype(ones(1)) .* v.reward_weight,
            atype(ones(1)) .* v.value_weight,
            dims=1
        )

        mse = mse_loss(pred_trajectory, joined_inputs, reduction="none") .* reshape(weights, (:, 1, 1))
        first_action_loss = v.first_action_weight .* mse_loss(
            joined_inputs[v.observation_dim+1:v.observation_dim+v.action_dim, 1, :], 
            pred_trajectory[v.observation_dim+1:v.observation_dim+v.action_dim, 1, :]
        )
        sum_reward_loss = v.sum_reward_weight .* mse_loss(
            mean(joined_inputs[end-1, :, :], dims=1), 
            mean(pred_trajectory[end-1, :, :], dims=1)
        )
        last_value_loss = v.last_value_weight .* mse_loss(
            joined_inputs[end, end, :], 
            pred_trajectory[end, end, :]
        )
        cross_entropy = binary_cross_entropy(pred_terminals, atype(clamp.(convert.(Float32, terminals),0.0f0, 1.0f0)))
        reconstruction_loss = mean((mse .* mask .* terminal_mask)) + cross_entropy
        reconstruction_loss = reconstruction_loss + first_action_loss + sum_reward_loss + last_value_loss

        if v.model.ma_update
            loss_vq = 0.0f0
        else
            loss_vq = mse_loss_detached(latents, feature)
        end
        loss_commit = mse_loss_detached(feature, latents)
    else
        reconstruction_loss = nothing
        loss_vq = nothing
        loss_commit = nothing
    end
    return reconstructed, reconstruction_loss, loss_vq, loss_commit

end


mutable struct TransformerPrior
    tok_emb::Embedding
    pos_emb::Param
    state_emb::Linear
    drop::Dropout
    blocks::Chain
    ln_f::LayerNorm
    head::Linear
    observation_dim
    vocab_size
    block_size
    embedding_dim

    function TransformerPrior(config)
        tok_emb = Embedding(config["n_embd"], config["K"])
        pos_emb = Param(atype(zeros(Float32, config["block_size"], config["n_embd"], 1)))
        state_emb = Linear(config["observation_dim"], config["n_embd"])
        drop = Dropout(config["embd_pdrop"])
        blocks = Chain([Block(config) for _ in 1:config["n_layer"]]...)
        ln_f = LayerNorm(config["n_embd"])
        head = Linear(config["n_embd"], config["K"]; bias=false)
        observation_dim = config["observation_dim"]
        vocab_size = config["K"]
        block_size = config["block_size"]
        embedding_dim = config["n_embd"]

        new(tok_emb, pos_emb, state_emb, drop, blocks, ln_f, head, observation_dim, vocab_size, block_size, embedding_dim)
    end
end

function (t::TransformerPrior)(idx, state, targets=nothing)
    if !(idx === nothing)
        t, b = size(idx)
        @assert t <= t.block_size  "Cannot forward, model block size is exhausted."
        token_embeddings = t.tok_emb(idx) # each index maps to a (learnable) vector
        token_embeddings = cat(atype(zeros(Float32, t.embedding_dim, 1, b)), token_embeddings, dims=2)
    else
        b = 1; t =0
        token_embeddings = atype(zeros(t.embedding_dim, 1, b))
    end

    ## [ embedding_dim x T+1 x 1 ]
    position_embeddings = t.pos_emb[:, 1:t+1, :]
    state_embeddings = reshape(t.state_emb(state), 1, :)
    ## [ embedding_dim x T+1 x 1 ]
    x = t.drop(token_embeddings .+ position_embeddings .+ state_embeddings)
    x = t.blocks(x)
    ## [ embedding_dim x T+1 x 1 ]
    x = t.ln_f(x)

    logits = t.head(x)
    logits = reshape(logits, (t.vocab_size, t+1, b))
    logits = logits[:, 1:t+1]

    if !(targets === nothing)
        logits = reshape(logits, t.vocab_size, :)
        targets = reshape(targets, :)
        loss = nll(logits, targets)
    else
        loss = nothing
    end
    return logits, loss
end
