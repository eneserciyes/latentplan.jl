using Test
using PyCall
using Knet
using Debugger: @enter, @bp, @run
using CUDA

if CUDA.functional()
	atype=KnetArray{Float32}
else	
	atype=Array{Float32}
end
cputype=Array{Float32}

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

vq_model = VQContinuousVAE(config)

### embed / Linear test ###
embed = Linear(config["transition_dim"], config["n_embd"])
embed.w = Param(atype(weights["model.embed.weight"][:cpu]()[:numpy]()))
embed.b = Param(atype(weights["model.embed.bias"][:cpu]()[:numpy]()))

# Reading input/output tensor
embed_test_input = atype(numpy.load("files/joined_inputs.npy"))
embed_test_gt = atype(numpy.load("files/token_embeddings.npy"));

@testset "Testing Embed Linear" begin
    embed_out = embed(permutedims(embed_test_input, (3, 2, 1)))
    eps = 5e-6

    @test all(abs.(cputype(embed_out .- permutedims(embed_test_gt, (3, 2, 1)))).<eps)

end;

### Block test ###
block = Block(config)
block.ln1.a = Param(atype(weights["model.encoder.0.ln1.weight"][:cpu]()[:numpy]()))
block.ln1.b = Param(atype(weights["model.encoder.0.ln1.bias"][:cpu]()[:numpy]()))
block.ln2.a = Param(atype(weights["model.encoder.0.ln2.weight"][:cpu]()[:numpy]()))
block.ln2.b = Param(atype(weights["model.encoder.0.ln2.bias"][:cpu]()[:numpy]()))

block.attn.key.w = Param(atype(weights["model.encoder.0.attn.key.weight"][:cpu]()[:numpy]()))
block.attn.key.b = Param(atype(weights["model.encoder.0.attn.key.bias"][:cpu]()[:numpy]()))
block.attn.query.w = Param(atype(weights["model.encoder.0.attn.query.weight"][:cpu]()[:numpy]()))
block.attn.query.b = Param(atype(weights["model.encoder.0.attn.query.bias"][:cpu]()[:numpy]()))
block.attn.value.w = Param(atype(weights["model.encoder.0.attn.value.weight"][:cpu]()[:numpy]()))
block.attn.value.b = Param(atype(weights["model.encoder.0.attn.value.bias"][:cpu]()[:numpy]()))
block.attn.proj.w = Param(atype(weights["model.encoder.0.attn.proj.weight"][:cpu]()[:numpy]()))
block.attn.proj.b = Param(atype(weights["model.encoder.0.attn.proj.bias"][:cpu]()[:numpy]()))

block.mlp.layers[1].w = Param(atype(weights["model.encoder.0.mlp.0.weight"][:cpu]()[:numpy]()))
block.mlp.layers[1].b = Param(atype(weights["model.encoder.0.mlp.0.bias"][:cpu]()[:numpy]()))
block.mlp.layers[3].w = Param(atype(weights["model.encoder.0.mlp.2.weight"][:cpu]()[:numpy]()))
block.mlp.layers[3].b = Param(atype(weights["model.encoder.0.mlp.2.bias"][:cpu]()[:numpy]()))

# Reading input/output tensor
block_test_input = atype(numpy.load("files/block_input.npy"))
block_test_gt = atype(numpy.load("files/block_output_gt.npy"));

@testset "Testing Block" begin
    block_out = block(permutedims(block_test_input, (3, 2, 1)))
    eps = 5e-6
    @test all(abs.(cputype(block_out .- permutedims(block_test_gt, (3, 2, 1)))).<eps)
end;

### Straigh Through test ### 

# Reading input/output tensor
st_input = atype(numpy.load("files/trajectory_feature.npy"))
latents_st_gt = atype(numpy.load("files/latents_st.npy"));
latents_gt = atype(numpy.load("files/latents.npy"));
ema_w_one_update_gt = atype(numpy.load("files/ema_w_one_update.npy"));
ema_count_one_update_gt = atype(numpy.load("files/ema_count_one_update.npy"));

codebook = VQEmbeddingMovingAverage(config["trajectory_embd"], config["K"])
codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
codebook.ema_count = Param(atype(weights["model.codebook.ema_count"][:cpu]()[:numpy]()))
codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))

@testset "Testing Straight Through Forward" begin
    latents_st, latents = straight_through(codebook, permutedims(st_input, (3, 2, 1)))
    eps = 5e-6
    @test all(abs.(cputype(latents_st .- permutedims(latents_st_gt, (3, 2, 1)))).<eps)
    @test all(abs.(cputype(latents .- permutedims(latents_gt, (3, 2, 1)))).<eps)
    @test all(abs.(cputype(codebook.ema_count .- ema_count_one_update_gt)).<eps)
    @test all(abs.(cputype(codebook.ema_w .- ema_w_one_update_gt')).<eps)
end;

### Encoder full test ###

vq_model.model.embed.w = Param(atype(weights["model.embed.weight"][:cpu]()[:numpy]()))
vq_model.model.embed.b = Param(atype(weights["model.embed.bias"][:cpu]()[:numpy]()))

vq_model.model.pos_emb = Param(atype(permutedims(weights["model.pos_emb"][:cpu]()[:numpy](), (3,2,1))))

for i in 1:config["n_layer"]
    vq_model.model.encoder.layers[i].ln1.a = Param(atype(weights["model.encoder.$(i-1).ln1.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].ln1.b = Param(atype(weights["model.encoder.$(i-1).ln1.bias"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].ln2.a = Param(atype(weights["model.encoder.$(i-1).ln2.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].ln2.b = Param(atype(weights["model.encoder.$(i-1).ln2.bias"][:cpu]()[:numpy]()))

    vq_model.model.encoder.layers[i].attn.key.w = Param(atype(weights["model.encoder.$(i-1).attn.key.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.key.b = Param(atype(weights["model.encoder.$(i-1).attn.key.bias"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.query.w = Param(atype(weights["model.encoder.$(i-1).attn.query.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.query.b = Param(atype(weights["model.encoder.$(i-1).attn.query.bias"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.value.w = Param(atype(weights["model.encoder.$(i-1).attn.value.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.value.b = Param(atype(weights["model.encoder.$(i-1).attn.value.bias"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.proj.w = Param(atype(weights["model.encoder.$(i-1).attn.proj.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].attn.proj.b = Param(atype(weights["model.encoder.$(i-1).attn.proj.bias"][:cpu]()[:numpy]()))

    vq_model.model.encoder.layers[i].mlp.layers[1].w = Param(atype(weights["model.encoder.$(i-1).mlp.0.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].mlp.layers[1].b = Param(atype(weights["model.encoder.$(i-1).mlp.0.bias"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].mlp.layers[3].w = Param(atype(weights["model.encoder.$(i-1).mlp.2.weight"][:cpu]()[:numpy]()))
    vq_model.model.encoder.layers[i].mlp.layers[3].b = Param(atype(weights["model.encoder.$(i-1).mlp.2.bias"][:cpu]()[:numpy]()))
end

vq_model.model.cast_embed.w = Param(atype(weights["model.cast_embed.weight"][:cpu]()[:numpy]()))
vq_model.model.cast_embed.b = Param(atype(weights["model.cast_embed.bias"][:cpu]()[:numpy]()))

# Reading input/output tensor
encoder_input = atype(numpy.load("files/joined_inputs.npy"))
encoder_test_gt = atype(numpy.load("files/trajectory_feature.npy"))

@testset "Testing Encoder" begin
    encoder_out = encode(vq_model.model, permutedims(encoder_input, (3, 2, 1)))
    eps = 5e-6
    @test all(abs.(cputype(encoder_out .- permutedims(encoder_test_gt, (3, 2, 1)))).<eps)
end;

### Decoder full test ###
vq_model.model.latent_mixing.w = Param(atype(weights["model.latent_mixing.weight"][:cpu]()[:numpy]()))
vq_model.model.latent_mixing.b = Param(atype(weights["model.latent_mixing.bias"][:cpu]()[:numpy]()))

for i in 1:config["n_layer"]
    vq_model.model.decoder.layers[i].ln1.a = Param(atype(weights["model.decoder.$(i-1).ln1.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].ln1.b = Param(atype(weights["model.decoder.$(i-1).ln1.bias"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].ln2.a = Param(atype(weights["model.decoder.$(i-1).ln2.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].ln2.b = Param(atype(weights["model.decoder.$(i-1).ln2.bias"][:cpu]()[:numpy]()))

    vq_model.model.decoder.layers[i].attn.key.w = Param(atype(weights["model.decoder.$(i-1).attn.key.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.key.b = Param(atype(weights["model.decoder.$(i-1).attn.key.bias"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.query.w = Param(atype(weights["model.decoder.$(i-1).attn.query.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.query.b = Param(atype(weights["model.decoder.$(i-1).attn.query.bias"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.value.w = Param(atype(weights["model.decoder.$(i-1).attn.value.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.value.b = Param(atype(weights["model.decoder.$(i-1).attn.value.bias"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.proj.w = Param(atype(weights["model.decoder.$(i-1).attn.proj.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].attn.proj.b = Param(atype(weights["model.decoder.$(i-1).attn.proj.bias"][:cpu]()[:numpy]()))

    vq_model.model.decoder.layers[i].mlp.layers[1].w = Param(atype(weights["model.decoder.$(i-1).mlp.0.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].mlp.layers[1].b = Param(atype(weights["model.decoder.$(i-1).mlp.0.bias"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].mlp.layers[3].w = Param(atype(weights["model.decoder.$(i-1).mlp.2.weight"][:cpu]()[:numpy]()))
    vq_model.model.decoder.layers[i].mlp.layers[3].b = Param(atype(weights["model.decoder.$(i-1).mlp.2.bias"][:cpu]()[:numpy]()))
end

vq_model.model.ln_f.a = Param(atype(weights["model.ln_f.weight"][:cpu]()[:numpy]()))
vq_model.model.ln_f.b = Param(atype(weights["model.ln_f.bias"][:cpu]()[:numpy]()))

vq_model.model.predict.w = Param(atype(weights["model.predict.weight"][:cpu]()[:numpy]()))
vq_model.model.predict.b = Param(atype(weights["model.predict.bias"][:cpu]()[:numpy]()))

# Reading input/output tensor
decoder_state_input = atype(numpy.load("files/state.npy"))
latents_st_input = atype(numpy.load("files/latents_st.npy"));
joined_pred_decoder_gt = atype(numpy.load("files/joined_pred.npy"));

@testset "Testing Decoder" begin
    decoder_out = decode(vq_model.model, permutedims(latents_st_input, (3,2,1)), decoder_state_input')
    eps = 5e-6
    @test all(abs.(cputype(decoder_out .- permutedims(joined_pred_decoder_gt, (3, 2, 1)))).<eps)
end;

### VQStepWiseTransformer full test ###

vq_model.model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
vq_model.model.codebook.ema_count = Param(atype(weights["model.codebook.ema_count"][:cpu]()[:numpy]()))
vq_model.model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))

# Reading input/output tensor
joined_inputs_input = atype(numpy.load("files/joined_inputs.npy"))
state_input = atype(numpy.load("files/state.npy"))
joined_pred_gt = atype(numpy.load("files/joined_pred.npy"))
latents_gt = atype(numpy.load("files/latents.npy"));
trajectory_feature_gt = atype(numpy.load("files/trajectory_feature.npy"))

@testset "Testing VQStepWiseTransformer" begin
    joined_pred, latents, trajectory_feature = vq_model.model(permutedims(joined_inputs_input, (3, 2, 1)), state_input')
    eps = 5e-6
    @test all(abs.(cputype(joined_pred .- permutedims(joined_pred_gt, (3, 2, 1)))).<eps)
    @test all(abs.(cputype(latents .- permutedims(latents_gt, (3, 2, 1)))).<eps)
    @test all(abs.(cputype(trajectory_feature .- permutedims(trajectory_feature_gt, (3, 2, 1)))).<eps)
end;


### VQContinuousVAE full test ###
padding_vector = atype(numpy.load("files/padding_vector.npy"))

vq_model.model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
vq_model.model.codebook.ema_count = Param(atype(weights["model.codebook.ema_count"][:cpu]()[:numpy]()))
vq_model.model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))
vq_model.padding_vector = padding_vector

# Reading input/output tensor
joined_inputs_input = atype(numpy.load("files/joined_inputs_vq_con_vae.npy"))
targets_input = atype(numpy.load("files/targets.npy"))
mask_input = atype(numpy.load("files/mask.npy"))
terminals_input = atype(numpy.load("files/terminals.npy"))
reconstructed_gt = atype(numpy.load("files/joined_pred.npy"))
reconstruction_loss_gt = atype(numpy.load("files/reconstruction_loss.npy"))
loss_vq_gt = 0
loss_commit_gt = atype(numpy.load("files/loss_commit.npy"))

@testset "Testing VQContinuousVAE" begin
    reconstructed, reconstruction_loss, loss_vq, loss_commit = vq_model(
        permutedims(joined_inputs_input, (3, 2, 1)), 
        permutedims(targets_input, (3,2,1)), 
        permutedims(mask_input, (3,2,1)), 
        permutedims(terminals_input, (3,2,1))
    )
    eps = 5e-6
    @test all(abs.(cputype(reconstructed .- permutedims(reconstructed_gt, (3, 2, 1)))).<eps)
    @test all(abs.(reconstruction_loss .- reconstruction_loss_gt.<eps))
    @test loss_vq == loss_vq_gt
    @test all(abs.(loss_commit .- loss_commit_gt.<eps))
end
