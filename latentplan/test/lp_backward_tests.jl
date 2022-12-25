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

function reset_codebook()
    vq_model.model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
    vq_model.model.codebook.ema_count = Param(atype(weights["model.codebook.ema_count"][:cpu]()[:numpy]()))
    vq_model.model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))
end

#######################
# setup vq_model weights

# encoder
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

# Decoder
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

# codebook
vq_model.model.codebook.embedding = Param(atype(weights["model.codebook.embedding"][:cpu]()[:numpy]()'))
vq_model.model.codebook.ema_count = Param(atype(weights["model.codebook.ema_count"][:cpu]()[:numpy]()))
vq_model.model.codebook.ema_w = Param(atype(weights["model.codebook.ema_w"][:cpu]()[:numpy]()'))

# padding vector
vq_model.padding_vector = atype(numpy.load("files/padding_vector.npy"))

println("Setup done..")
# Check forward first
@testset "Testing VQContinuousVAE forward" begin
    # Reading input/output tensor
    joined_inputs_input = atype(numpy.load("files/joined_inputs_vq_con_vae.npy"))
    targets_input = atype(numpy.load("files/targets.npy"))
    mask_input = atype(numpy.load("files/mask.npy"))
    terminals_input = atype(numpy.load("files/terminals.npy"))
    reconstructed_gt = atype(numpy.load("files/joined_pred.npy"))
    reconstruction_loss_gt = atype(numpy.load("files/reconstruction_loss.npy"))
    loss_vq_gt = 0
    loss_commit_gt = atype(numpy.load("files/loss_commit.npy"))

    reconstructed, reconstruction_loss, loss_vq, loss_commit = vq_model(
        permutedims(joined_inputs_input, (3, 2, 1)), 
        permutedims(targets_input, (3,2,1)), 
        permutedims(mask_input, (3,2,1)), 
        permutedims(terminals_input, (3,2,1))
    )

    eps = 5e-6
    @test all(abs.(cputype(reconstructed .- permutedims(reconstructed_gt, (3, 2, 1))).<eps))
    @test all(abs.(cputype(reconstruction_loss .- reconstruction_loss_gt).<eps))
    @test loss_vq == loss_vq_gt
    @test all(abs.(cputype(loss_commit .- loss_commit_gt).<eps))
end

# Testing straight through gradient
reset_codebook()
@testset "Testing straight through gradient" begin
    trajectory_feature_input = Param(atype(permutedims(numpy.load("files/trajectory_feature.npy"), (3,2,1))))
    traj_feat_grad_gt = atype(permutedims(numpy.load("files/grads/trajectory_feature_straight_through_grad.npy"), (3,2,1)))

    lossfunc = (x) -> 2 * x[1][1,1,1] ^ 2 + x[1][2,2,1] ^ 2 + x[1][3,3,1] ^ 2
    loss = @diff lossfunc(straight_through(vq_model.model.codebook, trajectory_feature_input))
    traj_feat_grad = grad(loss, trajectory_feature_input)

    eps = 5e-6
    @test all(abs.(cputype(traj_feat_grad .- traj_feat_grad_gt).<eps))
end

reset_codebook()
@testset "Testing straight through gradient complex" begin
    trajectory_feature_input = Param(atype(permutedims(numpy.load("files/trajectory_feature.npy"), (3,2,1))))
    traj_feat_grad_complex_gt = atype(permutedims(numpy.load("files/grads/traj_feat-after_decode-grad.npy"), (3,2,1)))
    state_gt = atype(numpy.load("files/state.npy")')

    lossfunc = (latents_st,latents,state) -> sum(decode(vq_model.model, latents_st, state))
    loss = @diff lossfunc(straight_through(vq_model.model.codebook, trajectory_feature_input)..., state_gt)
    traj_feat_grad = grad(loss, trajectory_feature_input)
    
    eps = 5e-5
    @test value(loss) ≈ 688.8791
    @test all(abs.(cputype(traj_feat_grad .- traj_feat_grad_complex_gt).<eps))
end

reset_codebook()
@testset "Testing cast_embed gradient complex" begin
    cast_embed_input = Param(atype(permutedims(numpy.load("files/cast_embed_input.npy"), (3,2,1))))
    cast_embed_grad_complex_gt = atype(numpy.load("files/grads/cast_embed_grad_complex.npy"))
    state_gt = atype(numpy.load("files/state.npy")')

    lossfunc = (latents_st,latents,state) -> sum(decode(vq_model.model, latents_st, state))
    loss = @diff lossfunc(straight_through(vq_model.model.codebook, vq_model.model.cast_embed(cast_embed_input))..., state_gt)
    cast_embed_grad_complex = grad(loss, vq_model.model.cast_embed.w)
    
    eps = 5e-3
    @test value(loss) ≈ 688.8791
    @test all(abs.(cputype(cast_embed_grad_complex .- cast_embed_grad_complex_gt).<eps))
end

reset_codebook()
@testset "Testing latent-mixing cat gradient" begin
    latents = Param(atype(permutedims(numpy.load("files/latents_st.npy"), (3,2,1))))
    state = atype(numpy.load("files/state.npy")')
    latent_mixing_input_grad_gt = atype(permutedims(numpy.load("files/grads/inputs_latent_mixing-grad.npy"), (3,2,1)))

    _, T, B = size(latents)
    loss = @diff sum(vq_model.model.latent_mixing(cat(repeat_broadcast(reshape(state, (:, 1, B)), 1, T, 1), latents, dims=1)))
    latents_grad = grad(loss, latents)

    eps = 5e-6
    @test value(loss) ≈ -23.2956
    @test all(abs.(cputype(latents_grad .- latent_mixing_input_grad_gt).<eps))
end

reset_codebook()
@testset "Testing decode gradient" begin
    latents_st_input = Param(atype(permutedims(numpy.load("files/latents_st.npy"), (3,2,1))))
    state_input = atype(numpy.load("files/state.npy")')
    latents_st_grad_gt = atype(permutedims(numpy.load("files/grads/latents_st-grad.npy"), (3,2,1)))
    
    loss = @diff sum(decode(vq_model.model, latents_st_input, state_input))
    latents_st_grad = grad(loss, latents_st_input)
    
    eps = 5e-5
    @test value(loss) ≈ 688.8791
    @test all(abs.(cputype(latents_st_grad .- latents_st_grad_gt).<eps))
end


reset_codebook()
# Testing gradients
losssum(prediction) = mean(prediction[2] + prediction[3] + prediction[4])
total_loss = @diff losssum(vq_model(
    atype(permutedims(joined_inputs_input, (3, 2, 1))), 
    atype(permutedims(targets_input, (3,2,1))), 
    atype(permutedims(mask_input, (3,2,1))), 
    atype(permutedims(terminals_input, (3,2,1)))
))

println("End-to-end gradient check..")

# check layer gradients

@testset "Checking predict gradients" begin
    ∇predict_w = grad(total_loss, vq_model.model.predict.w)
    ∇predict_b = grad(total_loss, vq_model.model.predict.b)
    ∇predict_w_gt = numpy.load("files/grads/predict-weight-grad.npy")
    eps = 5e-6
    @test all(abs.(cputype(∇predict_w) .- ∇predict_w_gt).<eps)
end;

@testset "Checking ln_f gradients" begin
    ∇ln_f_w = grad(total_loss, vq_model.model.ln_f.a)
    ∇ln_f_b = grad(total_loss, vq_model.model.ln_f.b)
    ∇ln_f_w_gt = numpy.load("files/grads/ln_f-weight-grad.npy")
    eps = 5e-6
    @test all(abs.(cputype(∇ln_f_w) .- ∇ln_f_w_gt).<eps)
end;

@testset "Checking latent_mixing gradients" begin
    ∇latent_mixing_w = grad(total_loss, vq_model.model.latent_mixing.w)
    ∇latent_mixing_b = grad(total_loss, vq_model.model.latent_mixing.b)
    ∇latent_mixing_w_gt = numpy.load("files/grads/latent_mixing-weight-grad.npy")
    eps = 5e-6
    @test all(abs.(cputype(∇latent_mixing_w) .- ∇latent_mixing_w_gt).<eps)
end;

@testset "Checking cast_embed gradients" begin
    ∇cast_embed_w_gt = numpy.load("files/grads/cast_embed-weight-grad.npy")

    ∇cast_embed_w = grad(total_loss, vq_model.model.cast_embed.w)
    ∇cast_embed_b = grad(total_loss, vq_model.model.cast_embed.b)
    
    eps = 5e-3
    @test all(abs.(cputype(∇cast_embed_w) .- ∇cast_embed_w_gt).<eps)
end;

@testset "Checking embed gradients" begin
    ∇embed_w_gt = numpy.load("files/grads/embed-weight-grad.npy")
    ∇embed_b_gt = numpy.load("files/grads/embed-bias-grad.npy")

    ∇embed_w = grad(total_loss, vq_model.model.embed.w)
    ∇embed_b = grad(total_loss, vq_model.model.embed.b)
    
    eps = 5e-3
    @test all(abs.(cputype(∇embed_w) .- ∇embed_w_gt).<eps)
    @test all(abs.(cputype(∇embed_b) .- ∇embed_b_gt).<eps)
end;


