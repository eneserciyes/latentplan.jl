include("datasets/sequence.jl")
include("utils/setup.jl")

using .Sequence: SequenceDataset
using .Setup: parser
using ArgParse: ArgParseSettings, @add_arg_table!, parse_args

s = ArgParseSettings()
@add_arg_table! s begin
    "--dataset"
        help = "which environment to use"
        arg_type = String
        default = "halfcheetah-medium-expert-v2"
    "--exp_name"
        help = "name of the experiment"
        arg_type = String
        default = "debug"
    "--tag"
        help = "any tag"
        arg_type = String
        default = "development"
    "--seed"
        help = "seed"
        arg_type = Int
        default = 42
    "--config"
        help = "relative jl file path with configurations"
        arg_type = String
        default = "../config/vqvae.jl"
end

super_args = parse_args(ARGS, s)
args = parser(super_args, experiment="train")


dataset = SequenceDataset(parsed_args["dataset"])
