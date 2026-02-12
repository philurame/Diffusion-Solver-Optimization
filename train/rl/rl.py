import warnings
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, Independent

class RLLoss(torch.nn.Module):
  def __init__(self, metric):
    super().__init__()
    self.metric = metric

  def forward(self, logprob_S, student, **kwargs):
    # losses: (S,B)
    losses = torch.stack([self.metric(s, **kwargs).detach() for s in student], dim=0).to(logprob_S.device)

    S = losses.shape[0]
    if S < 2:
      raise ValueError("RLOO requires num_samples >= 2")

    # baseline_loo: (S,B)
    baseline_loo = (losses.sum(dim=0, keepdim=True) - losses) / (S - 1)
    advantage = (losses - baseline_loo).detach()  # (S,B)
    rl_loss = (advantage * logprob_S[:, None]).mean(dim=0).sum()

    loss_mean_per_batch = losses.mean(dim=0)      # (B,)
    return rl_loss, loss_mean_per_batch.sum()


def rl_draw_params(
  num_samples: int,
  logits: torch.Tensor, # (2N,)
  max_resample: int = 200,
):
  alphas = F.softplus(logits)              # (2N,)
  dist = Independent(Dirichlet(alphas), 1) # event_dim=1

  # sample until all valid
  for _ in range(max_resample):
    flat = dist.sample((num_samples,)) # (S,2N)

    ts_len = flat.shape[1] // 2
    solver_timesteps    = flat[:,:ts_len]
    diffusion_timesteps = flat[:,ts_len:]

    if _validate_all_timesteps(solver_timesteps) and _validate_all_timesteps(diffusion_timesteps):
      logprob = dist.log_prob(flat) # 2N
      return solver_timesteps, diffusion_timesteps, logprob

  warnings.warn("rl_draw_params: giving up after max_resample attempts")
  logprob = dist.log_prob(flat)
  return solver_timesteps, diffusion_timesteps, logprob


def _validate_timesteps(timesteps, max_timestep=999.5, min_timestep=0, min_timestep_gap=0.15) -> bool:
  if timesteps is None or timesteps.numel() == 0:
    return True
  if (timesteps < min_timestep).any() or (timesteps > max_timestep).any():
    return False
  return (timesteps[:-1] - timesteps[1:] >= min_timestep_gap).all()
def _validate_all_timesteps(timesteps) -> bool:
  return all(_validate_timesteps(s) for s in timesteps)