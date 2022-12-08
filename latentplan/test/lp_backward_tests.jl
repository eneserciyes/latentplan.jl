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

vq_model = VQContinuousVAE(config)

function reset_codebook()
    vq_model.model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
    vq_model.model.codebook.ema_count = Param(weights["model.codebook.ema_count"][:cpu]()[:numpy]())
    vq_model.model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))
end

#######################
# setup vq_model weights

# encoder
vq_model.model.embed.w = Param(weights["model.embed.weight"][:cpu]()[:numpy]())
vq_model.model.embed.b = Param(weights["model.embed.bias"][:cpu]()[:numpy]())

vq_model.model.pos_emb = Param(permutedims(weights["model.pos_emb"][:cpu]()[:numpy](), (3,2,1)))

for i in 1:config["n_layer"]
    vq_model.model.encoder.layers[i].ln1.a = Param(weights["model.encoder.$(i-1).ln1.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].ln1.b = Param(weights["model.encoder.$(i-1).ln1.bias"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].ln2.a = Param(weights["model.encoder.$(i-1).ln2.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].ln2.b = Param(weights["model.encoder.$(i-1).ln2.bias"][:cpu]()[:numpy]())

    vq_model.model.encoder.layers[i].attn.key.w = Param(weights["model.encoder.$(i-1).attn.key.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.key.b = Param(weights["model.encoder.$(i-1).attn.key.bias"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.query.w = Param(weights["model.encoder.$(i-1).attn.query.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.query.b = Param(weights["model.encoder.$(i-1).attn.query.bias"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.value.w = Param(weights["model.encoder.$(i-1).attn.value.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.value.b = Param(weights["model.encoder.$(i-1).attn.value.bias"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.proj.w = Param(weights["model.encoder.$(i-1).attn.proj.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].attn.proj.b = Param(weights["model.encoder.$(i-1).attn.proj.bias"][:cpu]()[:numpy]())

    vq_model.model.encoder.layers[i].mlp.layers[1].w = Param(weights["model.encoder.$(i-1).mlp.0.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].mlp.layers[1].b = Param(weights["model.encoder.$(i-1).mlp.0.bias"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].mlp.layers[3].w = Param(weights["model.encoder.$(i-1).mlp.2.weight"][:cpu]()[:numpy]())
    vq_model.model.encoder.layers[i].mlp.layers[3].b = Param(weights["model.encoder.$(i-1).mlp.2.bias"][:cpu]()[:numpy]())
end

vq_model.model.cast_embed.w = Param(weights["model.cast_embed.weight"][:cpu]()[:numpy]())
vq_model.model.cast_embed.b = Param(weights["model.cast_embed.bias"][:cpu]()[:numpy]())

# Decoder
vq_model.model.latent_mixing.w = Param(weights["model.latent_mixing.weight"][:cpu]()[:numpy]())
vq_model.model.latent_mixing.b = Param(weights["model.latent_mixing.bias"][:cpu]()[:numpy]())

for i in 1:config["n_layer"]
    vq_model.model.decoder.layers[i].ln1.a = Param(weights["model.decoder.$(i-1).ln1.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].ln1.b = Param(weights["model.decoder.$(i-1).ln1.bias"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].ln2.a = Param(weights["model.decoder.$(i-1).ln2.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].ln2.b = Param(weights["model.decoder.$(i-1).ln2.bias"][:cpu]()[:numpy]())

    vq_model.model.decoder.layers[i].attn.key.w = Param(weights["model.decoder.$(i-1).attn.key.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.key.b = Param(weights["model.decoder.$(i-1).attn.key.bias"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.query.w = Param(weights["model.decoder.$(i-1).attn.query.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.query.b = Param(weights["model.decoder.$(i-1).attn.query.bias"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.value.w = Param(weights["model.decoder.$(i-1).attn.value.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.value.b = Param(weights["model.decoder.$(i-1).attn.value.bias"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.proj.w = Param(weights["model.decoder.$(i-1).attn.proj.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].attn.proj.b = Param(weights["model.decoder.$(i-1).attn.proj.bias"][:cpu]()[:numpy]())

    vq_model.model.decoder.layers[i].mlp.layers[1].w = Param(weights["model.decoder.$(i-1).mlp.0.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].mlp.layers[1].b = Param(weights["model.decoder.$(i-1).mlp.0.bias"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].mlp.layers[3].w = Param(weights["model.decoder.$(i-1).mlp.2.weight"][:cpu]()[:numpy]())
    vq_model.model.decoder.layers[i].mlp.layers[3].b = Param(weights["model.decoder.$(i-1).mlp.2.bias"][:cpu]()[:numpy]())
end

vq_model.model.ln_f.a = Param(weights["model.ln_f.weight"][:cpu]()[:numpy]())
vq_model.model.ln_f.b = Param(weights["model.ln_f.bias"][:cpu]()[:numpy]())

vq_model.model.predict.w = Param(weights["model.predict.weight"][:cpu]()[:numpy]())
vq_model.model.predict.b = Param(weights["model.predict.bias"][:cpu]()[:numpy]())

# codebook
vq_model.model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
vq_model.model.codebook.ema_count = Param(weights["model.codebook.ema_count"][:cpu]()[:numpy]())
vq_model.model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))

# padding vector
vq_model.padding_vector = atype(numpy.load("files/padding_vector.npy"))

# Reading input/output tensor
joined_inputs_input = numpy.load("files/joined_inputs_vq_con_vae.npy")
targets_input = numpy.load("files/targets.npy")
mask_input = numpy.load("files/mask.npy")
terminals_input = numpy.load("files/terminals.npy")
reconstructed_gt = numpy.load("files/joined_pred.npy")
reconstruction_loss_gt = numpy.load("files/reconstruction_loss.npy")
loss_vq_gt = 0
loss_commit_gt = numpy.load("files/loss_commit.npy")

println("Setup done..")

# # Check forward first
# @testset "Testing VQContinuousVAE forward" begin
#     reconstructed, reconstruction_loss, loss_vq, loss_commit = vq_model(
#         permutedims(joined_inputs_input, (3, 2, 1)), 
#         permutedims(targets_input, (3,2,1)), 
#         permutedims(mask_input, (3,2,1)), 
#         permutedims(terminals_input, (3,2,1))
#     )
#     eps = 5e-6
#     @test all(abs.(reconstructed .- permutedims(reconstructed_gt, (3, 2, 1))).<eps)
#     @test all(abs.(reconstruction_loss .- reconstruction_loss_gt).<eps)
#     @test loss_vq == loss_vq_gt
#     @test all(abs.(loss_commit .- loss_commit_gt).<eps)
# end


# Testing straight through gradient
reset_codebook()

trajectory_feature_input = Param(permutedims(numpy.load("files/trajectory_feature.npy"), (3,2,1)))
traj_feat_grad_gt = permutedims(numpy.load("files/trajectory_feature_straight_through_grad.npy"), (3,2,1))
@testset "Testing straight through gradient" begin
    lossfunc = (x) -> 2 * x[1][1,1,1] ^ 2 + x[1][2,2,1] ^ 2 + x[1][3,3,1] ^ 2
    loss = @diff lossfunc(straight_through(vq_model.model.codebook, trajectory_feature_input))
    traj_feat_grad = grad(loss, trajectory_feature_input)
    eps = 5e-6
    @test all(abs.(traj_feat_grad .- traj_feat_grad_gt).<eps)
end


# repeat gradient check


# Testing gradients
losssum(prediction) = begin println(typeof(prediction[4])); return mean(prediction[2] + prediction[3] + prediction[4]) end
total_loss = @diff losssum(vq_model(
    permutedims(joined_inputs_input, (3, 2, 1)), 
    permutedims(targets_input, (3,2,1)), 
    permutedims(mask_input, (3,2,1)), 
    permutedims(terminals_input, (3,2,1))
))

println("Gradient check..")
