import torch
import torch.optim as optim

def get_optimizer(config, sampler):
  train_params = []

  if sampler.st_train:
    train_params.append({"params": [sampler.solver_timesteps_logits], "lr": config.timesteps.lr})
  
  if sampler.dt_train:
    train_params.append({"params": [sampler.diffusion_timesteps_logits], "lr": config.timesteps.lr})
  
  if hasattr(sampler, "train_params"):
    train_params.append({"params": sampler.train_params, "lr": config.solver.lr})

  return optim.Adam(train_params)


def grad_clip(sampler, clip_grad_norm=1):
  log_dict = {}
  
  if sampler.st_train:
    log_dict['grads/solver_timesteps_absmax'] = sampler.solver_timesteps_logits.grad.abs().max().item()
    torch.nn.utils.clip_grad_norm_([sampler.solver_timesteps_logits], max_norm=clip_grad_norm)
  
  if sampler.dt_train:
    log_dict['grads/diffusion_timesteps_absmax'] = sampler.diffusion_timesteps_logits.grad.abs().max().item()
    torch.nn.utils.clip_grad_norm_([sampler.diffusion_timesteps_logits], max_norm=clip_grad_norm)

  if hasattr(sampler, "train_params"):
    log_dict['grads/solver_params_absmax'] = max([i.grad.abs().max().item() for i in sampler.train_params])
    torch.nn.utils.clip_grad_norm_(sampler.train_params, max_norm=clip_grad_norm)
  
  return log_dict