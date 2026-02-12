import torch
import torch.nn.functional as F
from .registries import solver_registry

# from torch.nn.utils import parameters_to_vector, vector_to_parameters
# def get_flat_params(self) -> torch.Tensor:
#   return parameters_to_vector(self.train_params)
# def set_flat_params(self, flat_params: torch.Tensor) -> None:
#   vector_to_parameters(flat_params, self.train_params)


@solver_registry.add_to_registry("SCALAR")
class SCALAR:
  '''
  x_prev = theor_outp + c_1(t) * x_t + c_2(t) * x_{t+1} + c_3(t) * x_{t+2}
  '''
  order = 3
  def _step(self, model_output, sample, theor_outp, step_index, **kwargs):
    self.train_model_outputs = [model_output] + self.train_model_outputs[:self.order-1] 
    num_params = min(self.order+1, step_index+2)
    features = [sample] + self.train_model_outputs

    train_deltas = self.train_params[0][step_index][:num_params]
    train_outp  = sum([i * j for i, j in zip(train_deltas, features)])

    prev_sample = theor_outp + train_outp
    return prev_sample
  
  def set_train_solver(self, sample, **kwargs):
    if getattr(self, "train_params", None) is None:
      self.train_params = torch.zeros(self.num_inference_steps, self.order+1, device=sample.device, dtype=sample.dtype, requires_grad=True)
      self.train_params = [self.train_params]
      self.train_model_outputs = []


@solver_registry.add_to_registry("CHANNEL")
class CHANNEL:
  '''
  x_prev = theor_outp + c_1(t, channel)...
  '''
  order = 3

  def _step(self, model_output, sample, theor_outp, step_index, **kwargs):
    self.train_model_outputs = [model_output] + self.train_model_outputs[:self.order - 1]
    num_params = min(self.order + 1, step_index + 2)
    features = [sample] + self.train_model_outputs

    train_deltas = self.train_params[0][step_index, :num_params]
    expand_shape = [1, -1] + [1] * (sample.ndim - 2)
    train_outp = 0.0
    for k in range(num_params):
      coeff = train_deltas[k].view(*expand_shape)  # to (1, C, 1, 1)
      train_outp = train_outp + coeff * features[k]

    prev_sample = theor_outp + train_outp
    return prev_sample

  def set_train_solver(self, sample, **kwargs):
    assert sample.ndim == 4
    C = sample.shape[1] # sample: (B, C, ...)
    if getattr(self, "train_params", None) is None:
      self.train_params = torch.zeros(self.num_inference_steps, self.order + 1, C, device=sample.device, dtype=sample.dtype, requires_grad=True)
      self.train_params = [self.train_params]
      self.train_model_outputs = []


@solver_registry.add_to_registry("CONV")
class CONV:
  """
  x_prev = theor_outp + c_1(t, DWConv)...
  """
  order = 3
  kernel_size = 3

  def _step(self, model_output, sample, theor_outp, step_index, **kwargs):
    self.train_model_outputs = [model_output] + self.train_model_outputs[:self.order - 1]
    num_params = min(self.order + 1, step_index + 2)
    features = [sample] + self.train_model_outputs

    train_outp = 0.0
    for k in range(num_params):
      weight = self.train_params[0][step_index, k]  # (C,1,K,K)
      bias = self.train_params[1][step_index, k] # (C,)
      conv = F.conv2d(features[k], weight=weight, bias=bias, padding=self.kernel_size//2, groups=sample.shape[1])
      train_outp = train_outp + conv

    prev_sample = theor_outp + train_outp
    return prev_sample
  
  def set_train_solver(self, sample, **kwargs):
    assert sample.ndim == 4
    C = sample.shape[1]
    K = self.kernel_size
    if getattr(self, "train_params", None) is None:
      # Per step, per term, per channel depthwise kernel (C, 1, K, K)
      train_params_w = torch.zeros(self.num_inference_steps, self.order + 1, C, 1, K, K, device=sample.device, dtype=sample.dtype, requires_grad=True)
      train_params_b = torch.zeros(self.num_inference_steps, self.order + 1, C, device=sample.device, dtype=sample.dtype, requires_grad=True)
      self.train_params = [train_params_w, train_params_b]
      self.train_model_outputs = []
