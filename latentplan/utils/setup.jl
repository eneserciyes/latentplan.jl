module Setup

using Configurations: to_dict
using Knet: seed!

export watch
function watch(args_to_watch::Vector{Tuple{String, String}})::Function
    function _fn(args)
        exp_name = []
        for (key, label) in args_to_watch
            if !hasproperty(args, key)
                continue
            end
            
            val = getfield(args, key)
            push!(exp_name, "$label$val")
        end
        exp_name = join(exp_name, "_")
        exp_name = replace(exp_name, "/_" => "/")
        
    end
    return _fn
end


export parser
function parser(args::Dict{String, Any}; experiment::Union{String, Nothing}=nothing)
    if !haskey(args, "config")
        return args
    end
    params = to_dict(read_config(args, experiment))

    args = merge(params, args)

    seed!(args["seed"])
    make_dir(args)
    
    args["task_type"] = "locomotion"
    args["obs_shape"] = [-1]
    if occursin("MineRL", args["dataset"])
        args["task_type"] = "MineRL"
        args["obs_shape"] = [3, 64, 64]
    elseif args["dataset"] in ["Breakout", "Pong", "Qbert", "Seaquest"]
        args["task_type"] = "atari"
        args["obs_shape"] = [4, 84, 84]
    end

    return args
end

function make_dir(args)
    if haskey(args, "logbase") && haskey(args, "dataset") && haskey(args, "exp_name")
        args["savepath"] = joinpath(expanduser(args["logbase"]), args["dataset"], args["exp_name"])
        if haskey(args, "suffix")
            args["savepath"] = joinpath(args["savepath"], args["suffix"])
        end
        try
            mkdir(args["savepath"])
        catch e
            println(args["savepath"], " already exists. Proceeding...")
        end
        println("Made directory", args["savepath"])
    end
end


function read_config(args::Dict{String, Any}, experiment::Union{String, Nothing})
    dataset = replace(args["dataset"], "-" => "_")
    config = args["config"]
    println("[ utils/setup ] Reading config: $config:$dataset")
    config_module = include(config)
    if hasproperty(config_module, Symbol(dataset))
        params = getproperty(config_module, Symbol(dataset))[experiment]
        print("Overriding base configs with $dataset configs.")
    else
        if experiment == "train"
            params = config_module.Train()
        elseif experiment == "plan"
            params = config_module.Plan()
        end
    end
    return params
end

end
