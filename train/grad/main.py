from setproctitle import setproctitle
setproctitle("@philurame (telegram)") 

import os, uuid, yaml, click, mlflow
from omegaconf import OmegaConf
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
os.environ['HF_HOME']    = os.path.join(ROOT, 'data', 'cache')
os.environ['TORCH_HOME'] = os.path.join(ROOT, 'data', 'cache')

# DETERMINISTIC
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True, warn_only=True)

import torch
torch.hub.set_dir(os.path.join(ROOT, 'data', 'cache'))

from .log_utils import log_config, install_mlflow_guards

def flatten_cfg(d, prefix=""):
  for k, v in d.items():
    key = f"{prefix}.{k}" if prefix else k
    if isinstance(v, dict):
      yield from flatten_cfg(v, key) # recursive
    else:
      yield key, v

@click.command()
@click.option("--config",  "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--device",  "-d", required=True, help="CUDA device ordinal, e.g. 0")
@click.option("--main_server", "-s", required=False, default=False, help="Is it MKUZ2?")
def main(config: str, device: str, main_server: bool):
  with open(config) as f:
    cfg_dict: dict = yaml.safe_load(f) 

  cfg_dict["device"] = f"cuda:{device}"

  exp_name = 'SD15'
  
  run_name = f"{cfg_dict['name']}_{uuid.uuid4().hex[:6]}"
  cfg_dict['run_id'] = run_name

  cfg = OmegaConf.create(cfg_dict) 

  cfg_dict = dict(flatten_cfg(cfg_dict))
  print(
    '\n'+'#'*50, 
    *[f'{k}={v}' for k, v in cfg_dict.items()],
    '#'*50+'\n', sep='\n', flush=True
  )
  tags = {
    "nfe": cfg.nfe,
    "loss": cfg.loss.name,
    "type": "grad",

    "st_train": cfg.timesteps.train_solver_timesteps,
    "dt_train": cfg.timesteps.train_diffusion_timesteps,
    "theor_solver": cfg.solver.theor_solver,
    "train_solver": cfg.solver.train_solver,
        
    "device": f"cuda:{device}"
  }
  tags = {k:str(v) for k,v in tags.items()}

  if main_server:
    mlflow.set_tracking_uri("http://127.0.0.1:5013")
  else:
    mlflow.set_tracking_uri("http://10.224.62.71:5013") # MKUZ2's hostname -I
  print("Current tracking URI:", mlflow.get_tracking_uri())
  
  mlflow.set_experiment(f"{exp_name}")
  
  from .trainer import Trainer  
  with mlflow.start_run(run_name=run_name, tags=tags, log_system_metrics=False) as run:
    install_mlflow_guards() # ends run if killed
    log_config(cfg_dict)
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
  import warnings
  warnings.filterwarnings('ignore')
  main()


