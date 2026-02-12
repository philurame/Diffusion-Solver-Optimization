import sys, os, random, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)

from lib.model import construct_pipeline
from lib.data import TrainDataset

from .losses  import loss_registry
from .log_utils import log_epoch, log_scalars
from .optim_utils import get_optimizer, grad_clip

import time, torch, tqdm, gc

def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False


class Trainer:

  def __init__(self, config):
    seed_everything()

    self.config = config

    self.train_dataset = TrainDataset('train', config, device=config.device)
    self.val_dataset   = TrainDataset('val',   config, device=config.device)

    theor_solver = config.solver.theor_solver
    train_solver = config.solver.train_solver
    self.pipe_kwargs = {
      'st_train': config.timesteps.train_solver_timesteps,
      'dt_train': config.timesteps.train_diffusion_timesteps,
      'timesteps_param_method': config.timesteps.param_method,
    }
    
    self.pipe = construct_pipeline(theor_solver, config.device, train_solver=train_solver, is_train=True)
    self.pipe('',num_inference_steps=config.nfe, dry_run=True, **self.pipe_kwargs)

    self.optimizer = get_optimizer(config, self.pipe.sampler)

    self.loss_fn = loss_registry[config.loss.name](device=config.device, hps_coef=config.loss.get('hps_coef'))

    self.global_step = 1
  

  def train(self):
    for epoch in tqdm.tqdm(range(self.config.epochs)):
      # train
      time_start  = time.perf_counter()
      train_loss, train_loss_logs, train_imgs_log = self.train_epoch()
      time_finish = time.perf_counter()
      gc.collect()
      torch.cuda.empty_cache()

      # validate
      val_loss,   val_loss_logs,   val_imgs_log   = self.validate_epoch()
      gc.collect()
      torch.cuda.empty_cache()
      
      log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_loss_logs': train_loss_logs,
        'train_imgs_log': train_imgs_log, 
        'val_loss': val_loss,
        'val_loss_logs': val_loss_logs,
        'val_imgs_log': val_imgs_log, 
        'time_train': time_finish - time_start,
        'global_step': self.global_step,
      }

      log_epoch(log_data, self.pipe.sampler, self.config)
  
      
      

  def train_epoch(self):
    with torch.autograd.detect_anomaly(check_nan=True):

      train_loss = 0.
      train_size      = self.config.dataset.train_size
      batch_size      = self.config.dataset.train_batch_size
      mini_batch_size = self.config.dataset.train_mini_batch_size

      n_imgs_log = int((min(train_size, 9))**0.5) ** 2 if self.config.get('log_imgs', False) else 0
      train_imgs_log = {}

      # batch
      for batch_start in range(0, train_size, batch_size):
        _batch_size = min(batch_size, train_size - batch_start)
        if _batch_size <= 0: continue
        batch_loss = 0.

        # mini-batch
        for mini_batch_start in range(0, _batch_size, mini_batch_size):
          _mini_batch_size = min(mini_batch_size, _batch_size - mini_batch_start)
          start_idx = batch_start + mini_batch_start
          end_idx = start_idx + _mini_batch_size

          data = self.train_dataset[start_idx:end_idx]
          
          student = self.pipe(
            prompt=data['prompts'], 
            noise=data['noise'],
            output_type='pt',
            **self.pipe_kwargs
          )
          for p, st, tch in zip(data['prompts'], student, data['imgs']):
            if p in train_imgs_log or len(train_imgs_log) < n_imgs_log:
              train_imgs_log[p] = (st.squeeze().cpu().float(), tch.squeeze().cpu().float())

          loss = self.loss_fn(
            student = student,
            teacher = data['imgs'],
            prompts = data['prompts']
          ).sum()
          (loss/_batch_size).backward()

          batch_loss += loss.item()
          train_loss += loss.item()

        log_dict = grad_clip(self.pipe.sampler)

        # log batch stats
        # ======================================================
        log_dict[f'train/batch_{self.config.loss.name}'] = batch_loss/_batch_size
        if hasattr(self.loss_fn, 'logs'):
          for k, (v, n) in self.loss_fn.logs.items():
            log_dict[f'train/batch_{k}'] = v
        ts_dict = self.pipe.sampler.get_timesteps()
        if self.pipe.sampler.st_train:
          log_dict.update({f'params/batch_solver_timesteps_{n}': t.item() for n, t in enumerate(ts_dict['solver_timesteps'])})
        if self.pipe.sampler.dt_train:
          log_dict.update({f'params/batch_diffusion_timesteps_{n}': t.item() for n, t in enumerate(ts_dict['diffusion_timesteps'])})
        if hasattr(self.pipe.sampler, "train_params"):
          log_dict['params/batch_solver_absmax'] = max([i.abs().max().item() for i in self.pipe.sampler.train_params])
        log_scalars(log_dict, self.global_step)
        # ======================================================
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1
      
      train_loss /= train_size

      train_loss_logs = {}
      if hasattr(self.loss_fn, 'logs'):
        for k, (v, n) in self.loss_fn.logs.items():
          train_loss_logs[k] = v
          self.loss_fn.logs[k] = [0., 0]

    return train_loss, train_loss_logs, train_imgs_log
  

  @torch.inference_mode()
  def validate_epoch(self):
    seed_everything()

    val_size = min(self.config.dataset.val_size, len(self.val_dataset))
    batch_size = self.config.dataset.val_batch_size

    n_imgs_log = int((min(val_size, 9))**0.5) ** 2 if self.config.get('log_imgs', False) else 0
    val_imgs_log = {}
    
    val_loss = 0
    for start_idx in range(0,val_size,batch_size):
      data = self.val_dataset[start_idx:start_idx+batch_size]

      student = self.pipe(
        prompt=data['prompts'], 
        noise = data['noise'],
        output_type='pt',
        **self.pipe_kwargs
      )

      for p, st, tch in zip(data['prompts'], student, data['imgs']):
        if len(val_imgs_log) < n_imgs_log:
          val_imgs_log[p] = (st.squeeze().cpu().float(), tch.squeeze().cpu().float())

      val_loss += self.loss_fn(
        student = student,
        teacher = data['imgs'],
        prompts = data['prompts']
      ).sum()

    val_loss /= val_size

    val_loss_logs = {}
    if hasattr(self.loss_fn, 'logs'):
      for k, (v, n) in self.loss_fn.logs.items():
        val_loss_logs[k] = v
        self.loss_fn.logs[k] = [0., 0]

    return val_loss, val_loss_logs, val_imgs_log