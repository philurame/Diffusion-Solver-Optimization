import sys, os, random, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)

from lib.model import construct_pipeline
from lib.sampler import get_sampler
from lib.data import TrainDataset

from .rl import rl_draw_params, RLLoss
from .mp_infer import start_pool, stop_pool, map_jobs

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

    self.train_dataset = TrainDataset('train', config, device=config.device, dtype=torch.float16)
    self.val_dataset   = TrainDataset('val',   config, device=config.device, dtype=torch.float16)

    theor_solver = config.solver.theor_solver
    train_solver = config.solver.train_solver
    self.pipe_kwargs = {
      'st_train': config.timesteps.train_solver_timesteps,
      'dt_train': config.timesteps.train_diffusion_timesteps,
      'timesteps_param_method': "softplus",
    }

    devices = [int(d) for d in list(config.devices)]
    shared_cfg = {"theor_solver": theor_solver,"train_solver": train_solver}
    self.mp_pool = start_pool(devices, shared_cfg)

    self.sampler = get_sampler(theor_solver, train_solver)

    # MODEL SPECIFIC
    if hasattr(self.sampler, "set_train_solver"):
      latents = torch.randn(1, 4, 64, 64, device=config.device, dtype=torch.float16)
      self.sampler.set_train_solver(latents)

    self.optimizer = get_optimizer(config, self.sampler)

    self.loss_fn = loss_registry[config.loss.name](device=config.device, hps_coef=config.loss.get('hps_coef'))
    self.rl_loss_fn = RLLoss(self.loss_fn)

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
        batch_loss = raw_batch_loss = 0.

        all_jobs = []
        jobs_meta = []

        # mini-batch
        for mini_batch_start in range(0, _batch_size, mini_batch_size):
          _mini_batch_size = min(mini_batch_size, _batch_size - mini_batch_start)
          start_idx = batch_start + mini_batch_start
          end_idx = start_idx + _mini_batch_size

          data = self.train_dataset[start_idx:end_idx]

          logits = torch.cat([self.sampler.solver_timesteps_logits, self.pipe.sampler.diffusion_timesteps_logits], dim=-1)
          solver_timesteps, diffusion_timesteps, logprob = rl_draw_params(
            num_samples = self.config.num_samples,
            logits = logits,
          )

          start_job_id = len(all_jobs)
          for i in range(self.config.num_samples):
            all_jobs.append({
              "job_id": start_job_id + i,
              "prompts": data['prompts'],
              "noise": data['noise'].cpu(),
              "solver_timesteps": solver_timesteps[i].cpu(),
              "diffusion_timesteps": diffusion_timesteps[i].cpu(),
              "solver_params": [x.cpu() for x in self.sampler.train_params], # ??????
              "pipe_kwargs": self.pipe_kwargs
            })
          jobs_meta.append({
            "start_job_id": start_job_id,
            "data": data,
            "logprob": logprob,
          })
        

        batch_outputs = map_jobs(self.mp_pool, all_jobs)
        n = self.config.num_samples

        for meta in jobs_meta:
          start_id = meta["start_job_id"]
          data = meta["data"]
          logprob = meta["logprob"]

          student = [x.to(self.config.device) for x in batch_outputs[start_id:start_id+n]]
  
          if data['prompts'][0] in train_imgs_log or len(train_imgs_log) < n_imgs_log:
            train_imgs_log[data['prompts'][0]] = (
              student[0].squeeze().cpu().float(), data['imgs'][0].squeeze().cpu().float()
            )

          rl_loss, rl_loss_mean = self.rl_loss_fn(
            logprob,
            student=student,
            **data
          )
          raw_batch_loss = raw_batch_loss + rl_loss
          batch_loss += rl_loss_mean.item()
          train_loss += rl_loss_mean.item()

      # 5) Backprop and step
      (raw_batch_loss / _batch_size).backward()
      log_dict = grad_clip(self.sampler)

      # log batch stats
      # ======================================================
      log_dict[f'train/batch_{self.config.loss.name}'] = batch_loss/_batch_size
      if hasattr(self.loss_fn, 'logs'):
        for k, (v, n) in self.loss_fn.logs.items():
          log_dict[f'train/batch_{k}'] = v
      ts_dict = self.sampler.get_timesteps()
      if self.sampler.st_train:
        log_dict.update({f'params/batch_solver_timesteps_{n}': t.item() for n, t in enumerate(ts_dict['solver_timesteps'])})
      if self.sampler.dt_train:
        log_dict.update({f'params/batch_diffusion_timesteps_{n}': t.item() for n, t in enumerate(ts_dict['diffusion_timesteps'])})
      if hasattr(self.sampler, "train_params"):
        log_dict['params/batch_solver_absmax'] = max([i.abs().max().item() for i in self.sampler.train_params])
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