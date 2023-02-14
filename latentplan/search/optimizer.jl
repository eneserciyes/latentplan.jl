using PyCall

torch = @pyimport torch

function beam_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand, prob_threshold=0.05, likelihood_weight=5e2, prob_acc="product", return_info=false)
   contex = nothing
   state = x[1:prior.observation_dim, 1, :]
   acc_probs = atype(zeros(1))
   info = Dict();
   for step in 0:(steps÷model.latent_step)-1
      logits, _ = prior(contex, state)
      probs = softmax(logits, dims=1)
      log_probs = log.(probs)
      nb_samples = step==0 ? beam_width * n_expand : n_expand
      samples = torch.multinomial(probs, num_samples=nb_samples, replacement=true)
      samples_log_prob = cat([reshape(a[i, 0], size(a[i, 0])..., 1) for (a, i) in zip(log_probs, samples.+1)], dims=2)
      
      acc_probs = repeat_interleave(acc_probs, nb_samples) .+ reshape(samples_log_prob, :)
      contex = reshape(samples, step+1, :)

      prediction_raw = decode_from_indices(model, contex, state)
      prediction = reshape(prediction_raw, model.action_dim+model.observation_dim+3, :)

      r_t = prediction[end-2, :]
      V_t = prediction[end-1, :]
      if denormalize_rew !== nothing
         r_t = reshape(denormalize_rew(r_t), :, size(contex, ndims(contex)))
      end
      if denormalize_val !== nothing
         V_t = reshape(denormalize_val(V_t), :, size(contex, ndims(contex)))
      end

      discounts = cumprod(atype(ones(size(r_t)...) .* discount, dims=1))
      values = sum(r_t[1:end-1, :] .* discounts[1:end-1, :], dims=1) .+ V_t[end, :] .* discounts[end, :]

      likelihood_bonus = likelihood_weight .* clip(acc_probs, -1e5, log(prob_threshold)*(steps÷model.latent_step))
      nb_top = step < steps ÷ model.latent_step - 1 ? beam_width : 1
      
      values_with_b, index = torch.topk(values.+likelihood_bonus, nb_top)
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
      contex = contex[index]
      acc_probs = acc_probs[index]
   end
   optimal = prediction_raw[index[1]]
   print("predicted max value $(values[1])")
   if return_info
      return cputype(optimal), info
  else
      return cputype(optimal)
  end
end
