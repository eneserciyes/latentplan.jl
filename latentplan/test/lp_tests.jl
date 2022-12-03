using Test
using PyCall
using Knet
using Debugger: @enter, @bp, @run

atype=Array{Float32}

include("../models/common.jl")
include("../models/transformers.jl")
include("../models/vqvae.jl")
include("../setup.jl")

@pyimport torch

weights = torch.load("files/gpt_weights.pt")

@pyimport numpy

#######################
######## setup ########
#######################

super_args = Dict{String, Any}(
    "dataset"=> "halfcheetah-medium-expert-v2",
    "exp_name"=> "debug",
    "seed"=> 42,
    "config"=> "../config/vqvae.jl",
)

args = parser(super_args, experiment="train")

config = deepcopy(args)
config["block_size"] = 650
config["observation_dim"] = 17
config["action_dim"] = 6
config["transition_dim"] = 26
config["n_embd"] = args["n_embd"] * args["n_head"]
config["vocab_size"] = args["N"]

model = VQContinuousVAE(config).model

### embed / Linear test ###
embed = Linear(config["transition_dim"], config["n_embd"])
embed.w = Param(weights["model.embed.weight"][:cpu]()[:numpy]())
embed.b = Param(weights["model.embed.bias"][:cpu]()[:numpy]())

# Reading input/output tensor
embed_test_input = numpy.load("files/joined_inputs.npy")
embed_test_gt = numpy.load("files/token_embeddings.npy");

@testset "Testing Embed Linear" begin
    embed_out = embed(permutedims(embed_test_input, (3, 2, 1)))
    eps = 5e-6

    @test all(abs.(embed_out .- permutedims(embed_test_gt, (3, 2, 1))).<eps)

end;

### Block test ###
block = Block(config)
block.ln1.a = Param(weights["model.encoder.0.ln1.weight"][:cpu]()[:numpy]())
block.ln1.b = Param(weights["model.encoder.0.ln1.bias"][:cpu]()[:numpy]())
block.ln2.a = Param(weights["model.encoder.0.ln2.weight"][:cpu]()[:numpy]())
block.ln2.b = Param(weights["model.encoder.0.ln2.bias"][:cpu]()[:numpy]())

block.attn.key.w = Param(weights["model.encoder.0.attn.key.weight"][:cpu]()[:numpy]())
block.attn.key.b = Param(weights["model.encoder.0.attn.key.bias"][:cpu]()[:numpy]())
block.attn.query.w = Param(weights["model.encoder.0.attn.query.weight"][:cpu]()[:numpy]())
block.attn.query.b = Param(weights["model.encoder.0.attn.query.bias"][:cpu]()[:numpy]())
block.attn.value.w = Param(weights["model.encoder.0.attn.value.weight"][:cpu]()[:numpy]())
block.attn.value.b = Param(weights["model.encoder.0.attn.value.bias"][:cpu]()[:numpy]())
block.attn.proj.w = Param(weights["model.encoder.0.attn.proj.weight"][:cpu]()[:numpy]())
block.attn.proj.b = Param(weights["model.encoder.0.attn.proj.bias"][:cpu]()[:numpy]())

block.mlp.layers[1].w = Param(weights["model.encoder.0.mlp.0.weight"][:cpu]()[:numpy]())
block.mlp.layers[1].b = Param(weights["model.encoder.0.mlp.0.bias"][:cpu]()[:numpy]())
block.mlp.layers[3].w = Param(weights["model.encoder.0.mlp.2.weight"][:cpu]()[:numpy]())
block.mlp.layers[3].b = Param(weights["model.encoder.0.mlp.2.bias"][:cpu]()[:numpy]())

# Reading input/output tensor
block_test_input = numpy.load("files/block_input.npy")
block_test_gt = numpy.load("files/block_output_gt.npy");

@testset "Testing Block" begin
    block_out = block(permutedims(block_test_input, (3, 2, 1)))
    eps = 5e-6
    @test all(abs.(block_out .- permutedims(block_test_gt, (3, 2, 1))).<eps)
end;

### Straigh Through test ### 

# Reading input/output tensor
st_input = numpy.load("files/trajectory_feature.npy")
latents_st_gt = numpy.load("files/latents_st.npy");
latents_gt = numpy.load("files/latents.npy");

codebook = VQEmbeddingMovingAverage(config["trajectory_embd"], config["K"])
codebook.embedding = Param(weights["model.codebook.embedding"][:cpu]()[:numpy]()') .* 100
codebook.ema_count = Param(weights["model.codebook.ema_count"][:cpu]()[:numpy]())
codebook.ema_w = Param(weights["model.codebook.ema_w"][:cpu]()[:numpy]()') .* 100

@testset "Testing Straight Through Forward" begin
    latents_st, latents = straight_through(codebook, permutedims(st_input, (3, 2, 1)))
    eps = 5e-6
    @test all(abs.(latents_st .- permutedims(latents_st_gt, (3, 2, 1))).<eps)
    @test all(abs.(latents .- permutedims(latents_gt, (3, 2, 1))).<eps)
end;

### Encoder full test ###

model.embed.w = Param(weights["model.embed.weight"][:cpu]()[:numpy]())
model.embed.b = Param(weights["model.embed.bias"][:cpu]()[:numpy]())

model.pos_emb = Param(permutedims(weights["model.pos_emb"][:cpu]()[:numpy](), (3,2,1)))

for i in 1:config["n_layer"]
    model.encoder.layers[i].ln1.a = Param(weights["model.encoder.$(i-1).ln1.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].ln1.b = Param(weights["model.encoder.$(i-1).ln1.bias"][:cpu]()[:numpy]())
    model.encoder.layers[i].ln2.a = Param(weights["model.encoder.$(i-1).ln2.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].ln2.b = Param(weights["model.encoder.$(i-1).ln2.bias"][:cpu]()[:numpy]())

    model.encoder.layers[i].attn.key.w = Param(weights["model.encoder.$(i-1).attn.key.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.key.b = Param(weights["model.encoder.$(i-1).attn.key.bias"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.query.w = Param(weights["model.encoder.$(i-1).attn.query.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.query.b = Param(weights["model.encoder.$(i-1).attn.query.bias"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.value.w = Param(weights["model.encoder.$(i-1).attn.value.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.value.b = Param(weights["model.encoder.$(i-1).attn.value.bias"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.proj.w = Param(weights["model.encoder.$(i-1).attn.proj.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].attn.proj.b = Param(weights["model.encoder.$(i-1).attn.proj.bias"][:cpu]()[:numpy]())

    model.encoder.layers[i].mlp.layers[1].w = Param(weights["model.encoder.$(i-1).mlp.0.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].mlp.layers[1].b = Param(weights["model.encoder.$(i-1).mlp.0.bias"][:cpu]()[:numpy]())
    model.encoder.layers[i].mlp.layers[3].w = Param(weights["model.encoder.$(i-1).mlp.2.weight"][:cpu]()[:numpy]())
    model.encoder.layers[i].mlp.layers[3].b = Param(weights["model.encoder.$(i-1).mlp.2.bias"][:cpu]()[:numpy]())
end

model.cast_embed.w = Param(weights["model.cast_embed.weight"][:cpu]()[:numpy]())
model.cast_embed.b = Param(weights["model.cast_embed.bias"][:cpu]()[:numpy]())

# Reading input/output tensor
encoder_input = numpy.load("files/joined_inputs.npy")
encoder_test_gt = numpy.load("files/trajectory_feature.npy")

@testset "Testing Encoder" begin
    encoder_out = encode(model, permutedims(encoder_input, (3, 2, 1)))
    eps = 5e-6
    @test all(abs.(encoder_out .- permutedims(encoder_test_gt, (3, 2, 1))).<eps)
end;

### Decoder full test ###
model.latent_mixing.w = Param(weights["model.latent_mixing.weight"][:cpu]()[:numpy]())
model.latent_mixing.b = Param(weights["model.latent_mixing.bias"][:cpu]()[:numpy]())

for i in 1:config["n_layer"]
    model.decoder.layers[i].ln1.a = Param(weights["model.decoder.$(i-1).ln1.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].ln1.b = Param(weights["model.decoder.$(i-1).ln1.bias"][:cpu]()[:numpy]())
    model.decoder.layers[i].ln2.a = Param(weights["model.decoder.$(i-1).ln2.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].ln2.b = Param(weights["model.decoder.$(i-1).ln2.bias"][:cpu]()[:numpy]())

    model.decoder.layers[i].attn.key.w = Param(weights["model.decoder.$(i-1).attn.key.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.key.b = Param(weights["model.decoder.$(i-1).attn.key.bias"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.query.w = Param(weights["model.decoder.$(i-1).attn.query.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.query.b = Param(weights["model.decoder.$(i-1).attn.query.bias"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.value.w = Param(weights["model.decoder.$(i-1).attn.value.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.value.b = Param(weights["model.decoder.$(i-1).attn.value.bias"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.proj.w = Param(weights["model.decoder.$(i-1).attn.proj.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].attn.proj.b = Param(weights["model.decoder.$(i-1).attn.proj.bias"][:cpu]()[:numpy]())

    model.decoder.layers[i].mlp.layers[1].w = Param(weights["model.decoder.$(i-1).mlp.0.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].mlp.layers[1].b = Param(weights["model.decoder.$(i-1).mlp.0.bias"][:cpu]()[:numpy]())
    model.decoder.layers[i].mlp.layers[3].w = Param(weights["model.decoder.$(i-1).mlp.2.weight"][:cpu]()[:numpy]())
    model.decoder.layers[i].mlp.layers[3].b = Param(weights["model.decoder.$(i-1).mlp.2.bias"][:cpu]()[:numpy]())
end

model.ln_f.a = Param(weights["model.ln_f.weight"][:cpu]()[:numpy]())
model.ln_f.b = Param(weights["model.ln_f.bias"][:cpu]()[:numpy]())

model.predict.w = Param(weights["model.predict.weight"][:cpu]()[:numpy]())
model.predict.b = Param(weights["model.predict.bias"][:cpu]()[:numpy]())

# Reading input/output tensor
decoder_state_input = numpy.load("files/state.npy")
latents_st_input = numpy.load("files/latents_st.npy");
joined_pred_decoder_gt = numpy.load("files/joined_pred.npy")

@testset "Testing Decoder" begin
    decoder_out = decode(model, permutedims(latents_st_input, (3,2,1)), decoder_state_input')
    eps = 5e-6
    @test all(abs.(decoder_out .- permutedims(joined_pred_decoder_gt, (3, 2, 1))).<eps)
end;

### VQStepWiseTransformer full test ###

model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()')) .* 100
model.codebook.ema_count = Param(weights["model.codebook.ema_count"][:cpu]()[:numpy]())
model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()')) .* 100

# Reading input/output tensor
joined_inputs_input = numpy.load("files/joined_inputs.npy")
state_input = numpy.load("files/state.npy")
joined_pred_gt = numpy.load("files/joined_pred.npy")
latents_gt = numpy.load("files/latents.npy");
trajectory_feature_gt = numpy.load("files/trajectory_feature.npy")

@testset "Testing VQStepWiseTransformer" begin
    joined_pred, latents, trajectory_feature = model(permutedims(joined_inputs_input, (3, 2, 1)), state_input')
    eps = 5e-6
    println(joined_pred[1,:,1])
    println(permutedims(joined_pred_gt, (3, 2, 1))[1,:,1])
    println(latents[1,:,1])
    println(permutedims(latents_gt, (3, 2, 1))[1,:,1])
    @test all(abs.(joined_pred .- permutedims(joined_pred_gt, (3, 2, 1))).<eps)
    @test all(abs.(latents .- permutedims(latents_gt, (3, 2, 1))).<eps)
    @test all(abs.(trajectory_feature .- permutedims(trajectory_feature_gt, (3, 2, 1))).<eps)
end;
