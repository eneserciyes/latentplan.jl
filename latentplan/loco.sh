#!/usr/bin/bash

name=T-1
datasets=(hopper-medium-replay-v2)

foo() {
  local round=$1
  for data in ${datasets[@]}; do
    # julia --project=.. train.jl --dataset $data --exp_name $name-$round --tag development --seed $round
    julia --project=.. trainprior.jl --dataset $data --exp_name $name-$round
    # for i in {1..20};
    # do
    #   julia --project=. plan.jl --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --n_expand 4 --beam_width 256 --horizon 24
    # done
  done
}

for round in {42..42}; do
  foo "$round"
done
