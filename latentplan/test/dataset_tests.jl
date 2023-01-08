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
include("../setup.jl")

super_args = Dict{String, Any}(
    "dataset"=> "halfcheetah-medium-expert-v2",
    "exp_name"=> "debug",
    "seed"=> 42,
    "config"=> "../config/vqvae.jl",
)

args = parser(super_args, experiment="train")
env_name = occursin("-v", args["dataset"]) ? args["dataset"] : args["dataset"] * "-v0"

# env params
sequence_length = args["subsampled_sequence_length"] * args["step"]
args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])
if !isdir(args["savepath"])
    mkpath(args["savepath"])
end

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


@testset "Dataset properties" begin
    
    @test length(dataset) == 1997999
    @test dataset.train_portion == 1.0
    @test dataset.observation_dim == 17
    @test size(dataset.observations_raw) == (17, 1999999)
    
    @test all(dataset.act_mean .≈ [-0.15910947, -0.3120827 , -0.6037344 , -0.12721658, -0.24652228, -0.07721385])
    @test all(dataset.act_std .≈ [0.8562386 , 0.69939375, 0.6349345 , 0.7109362 , 0.7055399 , 0.67665577])
    @test all(dataset.obs_mean .≈ [-0.05667465,  0.02437006, -0.06167089, -0.22351524, -0.26751527,
    -0.07545713, -0.05809679, -0.02767492,  8.110625  , -0.06136343,
    -0.17987022,  0.25174642,  0.24186617,  0.2519294 ,  0.587967  ,
    -0.24090458, -0.03018233])
    @test all(dataset.obs_std .≈ [ 0.06103434,  0.36054006,  0.4554429 ,  0.38476795,  0.22183533,
    0.56675154,  0.31966737,  0.28529128,  3.4438207 ,  0.67281306,
    1.8616966 ,  9.575805  , 10.029895  ,  5.903439  , 12.128177  ,
    6.4811788 ,  6.378619  ])

    @test dataset.reward_mean ≈ 7.713381
    @test dataset.reward_std ≈ 3.4197202
    @test dataset.value_mean ≈ 718.5527
    @test dataset.value_std ≈ 334.07663

    @test dataset.step == 1
    @test dataset.sequence_length == 25
    @test length(dataset.path_lengths) == 2000
end
