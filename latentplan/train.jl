using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using Statistics: mean
using Printf
using Knet
using Debugger: @enter, @bp, @run
using PyCall: pyimport

# wandb = pyimport("wandb")
# only while debugging
# using JuliaInterpreter
# using MethodAnalysis
# visit(Base) do item
#     isa(item, Module) && push!(JuliaInterpretr.compiled_modules, item)
#     true
# end

include("LPCore.jl")
include("setup.jl")

losssum(prediction) = mean(prediction[2] + prediction[3] + prediction[4])

function zerograd_embedding(model::VQContinuousVAE)
    model.model.codebook.embedding = value(model.model.codebook.embedding)
    model.model.codebook.ema_count = value(model.model.codebook.ema_count)
    model.model.codebook.ema_w = value(model.model.codebook.ema_w)
end

function vq_train(config, model::VQContinuousVAE, dataset::SequenceDataset; n_epochs=1, log_freq=100)
    loader = DataLoader(dataset; shuffle=false, batch_size=config["batch_size"])
    losses = []
    for (it, batch) in enumerate(loader)
        X, Y, mask, terminal = atype(batch[1]), atype(batch[2]), atype(batch[3]), atype(batch[4])

        if config["lr_decay"]
            lr_mult = 1.0f0
            lr = config["learning_rate"] * lr_mult
            for p in paramlist(model)
                p.opt.lr = lr
            end
        else
            lr = config["learning_rate"]
        end

        # forward the model
        total_loss = @diff losssum(model(X, Y, mask, terminal))
        println("Loss:", value(total_loss))
        for p in paramlist(model)
            update!(p, grad(total_loss, p))
        end
        # if it % log_freq == 1
        #     summary = Dict(
        #         "reconstruction_loss" => value(recon_loss),
        #         "commit_loss" => value(commit_loss),
        #         "lr" => lr
        #     )
        #     println(
        #         @sprintf(
        #             "[ utils/training ] epoch %d [ %d / %d ] train reconstruction loss %.5f | train commit loss %.5f | lr %.5f",
        #             n_epochs,
        #             it-1,
        #             length(loader),
        #             value(recon_loss),
        #             value(commit_loss),
        #             lr,
        #         )
        #     )
        #     # wandb.log(summary, step=n_epochs * length(loader) + it - 1)
        # end
        zerograd_embedding(model)
        # GC.gc(true)
    end
end

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
    "--seed"
    help = "seed"
    default = 42
    "--config"
    help = "relative jl file path with configurations"
    arg_type = String
    default = "../config/vqvae.jl"
    "--tag"
    help = "tag for the experiment"
    arg_type = String
    default = "debug"
end

#######################
######## setup ########
#######################

super_args = parse_args(ARGS, s)
args = parser(super_args, experiment="train")

#######################
####### dataset #######
#######################

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

obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
if args["task_type"] == "locomotion"
    transition_dim = obs_dim + act_dim + 3
else
    transition_dim = 128 + act_dim + 3
end

block_size = args["subsampled_sequence_length"] * transition_dim # total number of dimensionalities for a maximum length sequence (T)

println(
    "Dataset size: $(length(dataset)) |
    Joined dim: $transition_dim
    observation: $obs_dim, action: $act_dim | Block size: $block_size"
)

#######################
######## model ########
#######################

model_config = deepcopy(args)
model_config["block_size"] = block_size
model_config["observation_dim"] = obs_dim
model_config["action_dim"] = act_dim
model_config["transition_dim"] = transition_dim
model_config["n_embd"] = args["n_embd"] * args["n_head"]
model_config["vocab_size"] = args["N"]


model = VQContinuousVAE(model_config)

model.padding_vector = normalize_joined_single(dataset, atype(zeros(Float32, model.transition_dim - 1)))

warmup_tokens = length(dataset) * block_size
final_tokens = 20 * warmup_tokens

# training config
trainer_config = Dict(
    "batch_size" => args["batch_size"],
    "learning_rate" => args["learning_rate"],
    "betas" => (0.9, 0.95),
    "weight_decay" => 0.1,
    "grad_norm_clip" => 1.0,
    "warmup_tokens" => warmup_tokens,
    "kl_warmup_tokens" => warmup_tokens * 10,
    "final_tokens" => final_tokens,
    "lr_decay" => args["lr_decay"],
)

#######################
###### main loop ######
#######################
# set optimizers
opt_decay = AdamW(lr=trainer_config["learning_rate"], beta1=trainer_config["betas"][1], beta2=trainer_config["betas"][2], gclip=trainer_config["grad_norm_clip"])
opt_no_decay = AdamW(lr=trainer_config["learning_rate"], beta1=trainer_config["betas"][1], beta2=trainer_config["betas"][2], gclip=trainer_config["grad_norm_clip"])

for p in paramlist_decay(model)
    p.opt = clone(opt_decay)
end
for p in paramlist_no_decay(model)
    p.opt = clone(opt_no_decay)
end

## scale number of epochs to keep number of updates constant
n_epochs = Int(floor((1e6 / length(dataset)) * args["n_epochs_ref"]))
save_freq = Int(floor(n_epochs / args["n_saves"]))
# wandb.init(project="latentplan.jl", config=args, tags=[args["exp_name"], args["tag"]])
# load from checkpoint
model = Knet.load(joinpath(args["savepath"], "state_0.jld2"), "model")
for epoch in 1:n_epochs
    logfile = open(joinpath(args["savepath"], "log2.txt"), "a")
    
    epoch_message = @sprintf("\nEpoch: %d / %d | %s | %s\n", epoch, n_epochs, env_name, args["exp_name"])
    println(epoch_message)
    println(logfile, epoch_message)

    loader = DataLoader(dataset; shuffle=true, batch_size=trainer_config["batch_size"])
    for (it, batch) in enumerate(loader)
        X, Y, mask, terminal = atype(batch[1]), atype(batch[2]), atype(batch[3]), atype(batch[4])
        # forward the model
        total_loss = @diff losssum(model(X, Y, mask, terminal))
        println("Loss #", it, ": ", value(total_loss))
        println(logfile, "Loss #", it, ": ", value(total_loss))
        prev_model = deepcopy(model)
        if isnan(value(total_loss))
            println(logfile, "NaN loss!!")
            Knet.save(joinpath(args["savepath"], "nan_model_2.jld2"),"prev_model", prev_model, "model", model, "batch", batch)
            return
        end
        for p in paramlist(model)
            update!(p, grad(total_loss, p))
        end
        zerograd_embedding(model)
        if it % 100 == 1
            message = @sprintf(
                "[ utils/training ] epoch %d [ %d / %d ] train loss %.5f",
                n_epochs,
                it-1,
                length(loader),
                value(total_loss),
            )
            println(message)
            println(logfile, message)
        end
        # GC.gc(true)
    end
    close(logfile)
    save_epoch = (epoch + 1) ÷ save_freq * save_freq
    statepath = joinpath(args["savepath"], "state_$save_epoch.jld2")
    Knet.save(statepath, "model", model)
    println("Saved model to $statepath")
end
