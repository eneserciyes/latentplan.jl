using Test
using PyCall
using Knet

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

# model = VQContinuousVAE(config).model

### embed / Linear test ###
embed = Linear(config["transition_dim"], config["n_embd"])
embed.w = Param(weights["model.embed.weight"][:cpu]()[:numpy]())
embed.b = Param(weights["model.embed.bias"][:cpu]()[:numpy]())

# Reading input/output tensor
embed_test_input = numpy.load("files/joined_inputs_embed.npy")
embed_test_gt = numpy.load("files/token_embeddings_gt.npy");

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
    println(block_out[1,:,1])
    println(permutedims(block_test_gt, (3, 2, 1))[1,:,1])
    @test all(abs.(block_out .- permutedims(block_test_gt, (3, 2, 1))).<eps)
end;


