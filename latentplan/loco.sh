#!/usr/bin/bash

name=T-1
datasets=(hopper-medium-replay-v2)

foo() {
  local round=$1
  for data in ${datasets[@]}; do
    # julia --project=.. train.jl --dataset $data --exp_name $name-$round --tag development --seed $round
    # julia --project=.. trainprior.jl --dataset $data --exp_name $name-$round
    for i in {2..20};
    do
      julia --project=.. plan.jl --dataset $data --exp_name $name-$round --suffix $i --n_expand 4 --beam_width 64 --horizon 15
    done
  done
}

for round in {42..42}; do
  foo "$round"
done
