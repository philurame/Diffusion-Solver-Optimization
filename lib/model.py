from .SD15 import SD15
from .sampler import get_sampler

def construct_pipeline(theor_solver, device, train_solver=None, is_train=False, local_files_only=True, pipe=None):
  sampler = get_sampler(theor_solver, train_solver)

  if pipe is None:
    pipe = SD15.from_pretrained(is_train=is_train, device=device, local_files_only=local_files_only)
    
  pipe.sampler = sampler
  return pipe