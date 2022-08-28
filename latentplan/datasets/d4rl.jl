module D4RL

using PyCall

d4rl = pyimport("d4rl")
gym = pyimport("gym")

export load_environment
function load_environment(name::String)
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    env
end

end
