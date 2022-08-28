include("datasets/sequence.jl")

using .Sequence: SequenceDataset

env = "halfcheetah-medium-expert-v2"

dataset = SequenceDataset(env)
