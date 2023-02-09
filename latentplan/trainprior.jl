using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using Statistics: mean
using Printf
using Knet
using Debugger: @enter, @bp, @run
using JSON

include("LPCore.jl")
include("setup.jl")
using .LPCore

function prior_train(config, representation::VQContinuousVAE, model::TransformerPrior, dataset::SequenceDataset; n_epochs=1, log_freq=100)
    # set optimizers
    opt_decay = AdamW(lr=config["learning_rate"], beta1=config["betas"][1], beta2=config["betas"][2], weight_decay=config["weight_decay"], gclip=config["grad_norm_clip"])
    opt_no_decay = AdamW(lr=config["learning_rate"], beta1=config["betas"][1], beta2=config["betas"][2], weight_decay=0.0, gclip=config["grad_norm_clip"])

    for p in paramlist_decay(model)
        p.opt = clone(opt_decay)
    end
    for p in paramlist_no_decay(model)
        p.opt = clone(opt_no_decay)
    end

    n_tokens = 0
    loader = DataLoader(dataset; shuffle=true, batch_size=config["batch_size"])

    for epoch in 1:n_epochs
        losses = []
        for (it, batch) in enumerate(loader)
            y = batch[end-1]
            n_tokens += prod(size(y))

            if n_tokens < config["warmup_tokens"]
                # linear warmup
                lr_mult = float(n_tokens) / float(max(1, config["warmup_tokens"]))
            else
                # cosine learning rate decay
                progress = float(n_tokens - config["warmup_tokens"]) / float(
                    max(1, config["final_tokens"] - config["warmup_tokens"])
                )
                lr_mult = max(0.1, 0.5 * (1.0 + cos(pi * progress)))
            end

            if config["lr_decay"]
                lr = config["learning_rate"] * lr_mult
                for p in paramlist(model)
                    p.opt.lr = lr
                end
            else
                lr = config["learning_rate"]
            end
            
            states = batch[1][1:model.observation_dim, 1, :]
            indices = encode(representation, batch[1], batch[end])
            # forward the model
            total_loss = @diff model(indices[1:end-1,:], states, indices)
            push!(losses, value(total_loss))
            for p in paramlist(model)
                update!(p, grad(total_loss, p))
            end

            # report progress
            if it % log_freq == 0
                @printf(
                    "[ utils/training ] epoch %d [%d / %d] train loss %.5f, lr %.3e\n",
                    epoch,
                    it,
                    len(loader)
                    value(total_loss),
                    lr,
                )
            end
        end
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
end

#######################
######## setup ########
#######################

super_args = parse_args(ARGS, s)
args = parser(super_args, experiment="plan")

args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])

env_name = occursin("-v", args["dataset"]) ? args["dataset"] : args["dataset"] * "-v0"

dataset_config = Knet.load(joinpath(args["savepath"], "dataset_config.jld2"), "config")

dataset = SequenceDataset(
    dataset_config["env_name"];
    penalty=dataset_config["penalty"],
    sequence_length=dataset_config["sequence_length"], 
    step=dataset_config["step"], 
    discount=dataset_config["discount"], 
    disable_goal=dataset_config["disable_goal"], 
    normalize_raw=dataset_config["normalize_raw"], 
    normalize_reward=dataset_config["normalize_reward"],
    max_path_length=dataset_config["max_path_length"],
    atype=dataset_config["atype"]
)

obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim+1

gpt_epoch = args["gpt_epoch"]
representation = Knet.load(joinpath(args["savepath"], "state_$gpt_epoch.jld2"))
# representation.padding_vector = normalize_joined_single(dataset, atype(zeros(Float32, representation.transition_dim-1)))

args = parser(super_args, experiment="train")
args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])
block_size = args.subsampled_sequence_length ÷ args.latent_step
obs_dim = dataset.observation_dim

model_config = deepcopy(args)
model_config["block_size"] = block_size
model_config["observation_dim"] = obs_dim
model_config["n_embd"] = args["n_embd"] * args["n_head"]

model = TransformerPrior(model_config)

warmup_tokens = length(dataset) * block_size
final_tokens = 20 * warmup_tokens

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
opt_decay = AdamW(lr=config["learning_rate"], beta1=config["betas"][1], beta2=config["betas"][2], weight_decay=config["weight_decay"], gclip=config["grad_norm_clip"])
opt_no_decay = AdamW(lr=config["learning_rate"], beta1=config["betas"][1], beta2=config["betas"][2], weight_decay=0.0, gclip=config["grad_norm_clip"])

for p in paramlist_decay(model)
    p.opt = clone(opt_decay)
end
for p in paramlist_no_decay(model)
    p.opt = clone(opt_no_decay)
end

## scale number of epochs to keep number of updates constant
n_epochs = Int(floor(1e6 / length(dataset) * args["n_epochs_ref"]))
save_freq = Int(floor(n_epochs / args["n_saves"]))


for epoch in 1:n_epochs
    logfile = open(joinpath(args["savepath"], "log-prior.txt"), "a")
    
    epoch_message = @sprintf("\nEpoch: %d / %d | %s | %s", epoch, n_epochs, env_name, args["exp_name"])
    println(epoch_message)
    println(logfile, epoch_message)
    
    loader = DataLoader(dataset; shuffle=true, batch_size=trainer_config["batch_size"])
    
    for (it, batch) in enumerate(loader)
        X, Y, mask, terminal = atype(batch[1]), atype(batch[2]), atype(batch[3]), atype(batch[4])
        
        states = X[1:model.observation_dim, 1, :]
        indices = encode(representation, X, terminal)
        
        total_loss = @diff model(indices[1:end-1,:], states, indices)
        
        if isnan(value(total_loss))
            println(logfile, "NaN loss!!")
            return
        end
        
        for p in paramlist(model)
            update!(p, grad(total_loss, p))
        end
        
        if it % 100 == 1
            message = @sprintf(
                "[ utils/training ] epoch %d [ %d / %d ] train loss %.5f",
                epoch,
                it-1,
                length(loader),
                value(total_loss),
            )
            println(message)
            println(logfile, message)
        end
    end
    close(logfile)
    
    save_epoch = (epoch + 1) ÷ save_freq * save_freq
    statepath = joinpath(args["savepath"], "prior_state_$save_epoch.jld2")
    Knet.save(statepath, "model", model)
    println("Saved model to $statepath")
end
