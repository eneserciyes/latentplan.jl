module Sequence
include("d4rl.jl")

using .D4RL
using Printf

struct SequenceDataset;
    env;
    sequence_length;
    step;
    max_path_length;
    device;
    disable_goal;
    # normalized_raw;
    # normalize_reward;
    # obs_mean; obs_std;
    # act_mean; act_std;
    # reward_mean; reward_std;
    # observations_raw;
    # actions_raw;
    # joined_raw;
    # rewards_raw;
    # terminals_raw;
    # joined_segmented; termin
    function SequenceDataset(env, sequence_length::Int=250, step::Int=10, 
        discount::Float64=0.99, max_path_length::Int=1000,
        penalty=nothing, device::String="cuda:0", normalize_raw::Bool=true, normalize_reward::Bool=true,
        train_portion::Float64=1.0, disable_goal::Bool=false)
    
        @printf("[ datasets/sequence ] Sequence length: %d | Step: %d | Max path length: %d\n", sequence_length, step, max_path_length)
        
        env = typeof(env) == String ? load_environment(env) : env
        new(env, sequence_length, step, max_path_length, device, disable_goal)
    end
end



end
