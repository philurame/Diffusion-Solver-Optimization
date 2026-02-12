import torch
from .registries import solver_registry


@solver_registry.add_to_registry("DDIM")
class DDIM:
  order = 1
  def step(self, model_output, sample=None, **kwargs):
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    prev_sample = (alpha_t / alpha_s) * sample + model_output * alpha_t * (sigma_t - sigma_s)
    self.step_index += 1
    return prev_sample
  
  def sigma_to_alpha_t(self, sigma): # returns sqrt(alpha_t)
    return  1 / ((sigma**2 + 1) ** 0.5)
  

@solver_registry.add_to_registry('DEIS')
class DEIS:
  order = 2
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = self.model_outputs[-1:] + [model_output]
    sample = sample.to(torch.float32)
    
    if self.step_index == 0 or self.step_index == len(self.solver_timesteps) - 1:
      prev_sample = self.first_order_update(model_output, sample=sample)
    else:
      prev_sample = self.second_order_update(sample=sample)

    self.step_index += 1
    return prev_sample.to(model_output.dtype)
    
  def first_order_update(self, model_output, sample):
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, alpha_s = self.sigma_to_alpha_t(sigma_t), self.sigma_to_alpha_t(sigma_s)
    prev_sample = (alpha_t / alpha_s) * sample - model_output * alpha_t * (sigma_s - sigma_t)
    return prev_sample

  def second_order_update(self, sample):
    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[self.step_index + 1],
      self.sigmas[self.step_index],
      self.sigmas[self.step_index - 1],
    )
    rho_t, rho_s0, rho_s1 = sigma_t, sigma_s0, sigma_s1

    alpha_t, alpha_s0 = (
      self.sigma_to_alpha_t(sigma_t),
      self.sigma_to_alpha_t(sigma_s0),
    )
    m0, m1 = self.model_outputs[-1], self.model_outputs[-2]

    def ind_fn(t, b, c):
      return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

    coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
    coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)

    x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
    return x_t
  
  def sigma_to_alpha_t(self, sigma):
    return  1 / ((sigma**2 + 1) ** 0.5)