import torch
import torch.nn.functional as F

# from .registries import ClassRegistry
from .base_solvers import *
from .train_solvers import *
from .registries import solver_registry

def get_sampler(theor_solver=None, train_solver=None, **kwargs):
  if train_solver is not None:
    cls_theor_solver = solver_registry[theor_solver]
    cls_train_solver = solver_registry[train_solver]
    class ScheduleSolverClass(TimestepSampler, cls_theor_solver, cls_train_solver):
      def step(self, model_output, sample=None, **kwargs_):
        step_index = self.step_index
        theor_pred = super().step(model_output, sample, **kwargs_)
        res_pred = super()._step(model_output, sample, theor_pred, step_index, **kwargs_)
        return res_pred
  else:
    class ScheduleSolverClass(TimestepSampler, solver_registry[theor_solver]):
      pass
   
  return ScheduleSolverClass(**kwargs)


class TimestepSampler:
  def __init__(self, beta_start=0.00085, beta_end=0.012, **kwargs):
    self.num_train_timesteps = 1000

    # "beta_schedule": "scaled_linear"
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2
    self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    self.step_index = 0


  def set_timesteps(self, num_inference_steps=None, solver_timesteps=None, device=None, **kwargs): 
    self.step_index = 0
    self.model_outputs = []

    if solver_timesteps is None and getattr(self, 'solver_timesteps_logits', None) is not None: 
      solver_timesteps = self.get_timesteps(**kwargs)['solver_timesteps']
      self.sigmas = self._timesteps_to_sigmas(solver_timesteps, kwargs.get("last_sigma_zero", False))
      return 

    if solver_timesteps is None: # initialize LINEAR_SCHEDULE
      solver_timesteps = torch.linspace(self.num_train_timesteps - 1, 0, steps=num_inference_steps + 1, device=device).round()[:-1].float().to(device)
    else:
      if not isinstance(solver_timesteps, torch.Tensor): solver_timesteps = torch.tensor([i for i in solver_timesteps])
      solver_timesteps = solver_timesteps.float().to(device)
    
    self.solver_timesteps    = solver_timesteps
    self.diffusion_timesteps = solver_timesteps
    
    self.num_inference_steps = len(solver_timesteps)
    self.sigmas = self._timesteps_to_sigmas(solver_timesteps, kwargs.get("last_sigma_zero", False))

    # set train_timesteps
    self.st_train = kwargs.get("st_train", False)
    self.dt_train = kwargs.get("dt_train", False)
    if self.st_train or self.dt_train:
      self.max_timestep = kwargs.get("max_timestep", 999.5)
      self.timesteps_param_method = kwargs.get("timesteps_param_method", 'cumprod')

      logits = self._ts_to_logits(solver_timesteps)
      self.solver_timesteps_logits = torch.nn.Parameter(logits.clone(), requires_grad=self.st_train)
      self.diffusion_timesteps_logits = torch.nn.Parameter(logits.clone(), requires_grad=self.dt_train)
  

  def get_timesteps(self, **kwargs):
    solver_timesteps    = self.solver_timesteps
    diffusion_timesteps = self.diffusion_timesteps
    if self.st_train or self.dt_train:
      if self.timesteps_param_method == 'cumprod':
        solver_timesteps = self.max_timestep * torch.cumprod(F.sigmoid(self.solver_timesteps_logits), 0)
        diffusion_timesteps = self.max_timestep * torch.cumprod(F.sigmoid(self.diffusion_timesteps_logits), 0)
      if self.timesteps_param_method == 'softplus':
        cum_probs = torch.cumsum(F.softplus(self.solver_timesteps_logits), dim=0)
        cum_probs = cum_probs / cum_probs[-1].clamp_min(1e-8)
        solver_timesteps = (self.max_timestep - cum_probs * self.max_timestep)[:-1]
        cum_probs = torch.cumsum(F.softplus(self.diffusion_timesteps_logits), dim=0)
        cum_probs = cum_probs / cum_probs[-1].clamp_min(1e-8)
        diffusion_timesteps = (self.max_timestep - cum_probs * self.max_timestep)[:-1]

    return {
      "solver_timesteps": solver_timesteps,
      "diffusion_timesteps": diffusion_timesteps
    }


  def _ts_to_logits(self, timesteps):
    if isinstance(timesteps, list): timesteps = torch.tensor(timesteps)

    if self.timesteps_param_method == 'cumprod':
      logits = timesteps.clone() / self.max_timestep
      for i in range(1, len(timesteps)):
        logits[i] = max(timesteps[i] / timesteps[i-1], 1e-4)
      logits = torch.log(logits) - torch.log(1 - logits)
      return logits

    if self.timesteps_param_method == 'softplus':
      logits = torch.cat([(self.max_timestep - timesteps) / self.max_timestep, torch.tensor([1.], device=timesteps.device, dtype=timesteps.dtype)]).clone()
      logits = torch.cat([logits[:1], logits[1:] - logits[:-1]], dim=0) # logits[1:] = logits[1:] - logits[:-1]
      logits = torch.log(torch.exp(logits) - 1.0)
      return logits


  def _timesteps_to_sigmas(self, timesteps, last_sigma_zero):
    sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod)**0.5
    timesteps = timesteps.to(sigmas.device)
    N = sigmas.shape[0]

    idx_lower = timesteps.floor().long()
    idx_upper = idx_lower + 1
    idx_upper = torch.where(idx_upper >= N, idx_lower, idx_upper)
    w = timesteps - idx_lower.to(timesteps.dtype)
    sigma_interp = sigmas[idx_lower] * (1 - w) + sigmas[idx_upper] * w

    if timesteps[-1] < 0.001 or last_sigma_zero: 
      sigma_last = torch.tensor([0.], dtype=torch.float32)
    else:
      sigma_last = (((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0])**0.5).unsqueeze(0)
    sigmas = torch.cat([sigma_interp, sigma_last]).to(torch.float32)
    return sigmas