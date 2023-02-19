using PyCall
@pyimport torch

function beam_with_prior(prior, model, x, dataset; discount, steps,
                    beam_width, n_expand, prob_threshold=0.05, likelihood_weight=5e2, return_info=false)
   contex = nothing
   state = x[1:prior.observation_dim, 1, :]
   acc_probs = atype(zeros(1))
   info = Dict();
   index = nothing
   prediction_raw=nothing
   values=nothing
   for step in 0:(steps÷model.latent_step)-1
      logits, _ = prior(contex, state)
      probs = softmax(logits[:, end, :], dims=1)
      log_probs = log.(probs)
      nb_samples = step==0 ? beam_width * n_expand : n_expand
      samples = torch.multinomial(torch.tensor(cputype(probs)'), num_samples=nb_samples, replacement=true).numpy()' .+ 1
      samples_log_prob = atype(cat([reshape(a[i], size(a[i])..., 1) for (a, i) in zip(eachslice(cputype(log_probs), dims=2), eachslice(samples, dims=2))]..., dims=1))

      acc_probs = repeat_interleave(acc_probs, nb_samples) .+ reshape(samples_log_prob, :)
      if contex !== nothing
         contex = cat(repeat_interleave(contex, 1, nb_samples, t=Array{eltype(contex)}), reshape(samples, 1, :); dims=1)
      else
         contex = reshape(samples, step+1, :)
      end
      prediction_raw = decode_from_indices(model, contex, state)
      prediction = reshape(prediction_raw, model.action_dim+model.observation_dim+3, :)
      r_t = prediction[end-2, :]
      V_t = prediction[end-1, :]
      if dataset !== nothing
         r_t = reshape(denormalize_rewards(dataset, r_t), :, size(contex, ndims(contex)))
      end
      if dataset !== nothing
         V_t = reshape(denormalize_values(dataset, V_t), :, size(contex, ndims(contex)))
      end

      discounts = atype(cumprod(ones(size(r_t)...) .* discount, dims=1))
      values = dropdims_n(sum(r_t[1:end-1, :] .* discounts[1:end-1, :], dims=1), dims=(1,)) .+ V_t[end, :] .* discounts[end, :]

      likelihood_bonus = likelihood_weight .* clip(acc_probs, -1e5, log(prob_threshold)*(steps÷model.latent_step))
      nb_top = step < steps ÷ model.latent_step - 1 ? beam_width : 1
      
      values_with_b, index = torch.topk(torch.tensor(cputype(values.+likelihood_bonus)), nb_top)
      values_with_b = values_with_b.numpy()
      index = index.numpy()
      index.+=1
      if return_info
         info[(step+1)*model.latent_step] = Dict(
            "predictions"=>cputype(prediction_raw),
            "returns"=>cputype(values),
            "latent_codes"=>cputype(contex),
            "log_probs"=>cputype(acc_probs),
            "objectives"=>cputype(values.+likelihood_bonus),
            "index"=>cputype(index),
            )
      end
      contex = contex[:, index]
      acc_probs = acc_probs[index]
   end
   optimal = prediction_raw[:,:,index[1]]
   println("predicted max value $(values[1])")
   if return_info
      return cputype(optimal), info
   else
      return cputype(optimal)
  end
end
