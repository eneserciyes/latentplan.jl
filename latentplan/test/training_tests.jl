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

include("../datasets/sequence.jl")
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

# setup dataloader

env_name = occursin("-v", args["dataset"]) ? args["dataset"] : args["dataset"] * "-v0"

# env params
sequence_length = args["subsampled_sequence_length"] * args["step"]
args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])
if !isdir(args["savepath"])
    mkpath(args["savepath"])
end

println("Loading dataset..")

dataset = SequenceDataset(
    env_name;
    penalty=args["termination_penalty"], 
    sequence_length=sequence_length, 
    step=args["step"], 
    discount=args["discount"], 
    disable_goal=args["disable_goal"], 
    normalize_raw=args["normalize"], 
    normalize_reward=args["normalize_reward"],
    max_path_length=args["max_path_length"],
    atype=atype
)

loader = DataLoader(dataset; shuffle=false, batch_size=args["batch_size"])

println("Setup done..")

losssum(prediction) = mean(prediction[2] + prediction[3] + prediction[4])

opt_decay = AdamW(lr=args["learning_rate"], beta1=0.9, beta2=0.95, weight_decay=0.1, gclip=1.0)
opt_no_decay = AdamW(lr=args["learning_rate"], beta1=0.9, beta2=0.95, weight_decay=0.0, gclip=1.0)
for p in paramlist_decay(vq_model)
    p.opt = clone(opt_decay)
end
for p in paramlist_no_decay(vq_model)
    p.opt = clone(opt_no_decay)
end

@testset "Checking weights after 1 step for a single layer" begin
    grad_decoder_gt = atype(numpy.load("files/training/decoder0-attn-key-grad.npy"))

    for (it, batch) in enumerate(loader)
        total_loss = @diff losssum(vq_model(batch...))
        break
    end
    @test value(total_loss) .≈ 3.4647
    grad_decoder = grad(total_loss, vq_model.model.decoder.layers[1].attn.key.w)
    @test all(grad_decoder .≈ grad_decoder_gt)

    update!(vq_model.model.decoder.layers[1].attn.key.w, grad_decoder)
end
