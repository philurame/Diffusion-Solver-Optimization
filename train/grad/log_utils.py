import torch, mlflow
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import json, math, tempfile
from pathlib import Path
from typing import Dict, Any, List

import signal, atexit, os, sys

def log_scalars(log_dict: dict, step: int):
  scalars = {}
  for k, v in log_dict.items():
    s = _to_num(v)
    if s is not None and math.isfinite(s):
      scalars[k] = s
  if scalars:
    mlflow.log_metrics(scalars, step=step)

def log_list(name: str, data: List[float], artifact_dir: str = "artifacts/lists") -> None:
  """Log a list of floats as JSON artifact."""
  with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / f"{name}.json"
    p.write_text(json.dumps(data, indent=2))
    mlflow.log_artifact(str(p), artifact_path=artifact_dir)


def log_epoch(log_data, sampler, config):
  '''train_loss, val_loss, val_imgs_log, time_train, epoch, timesteps_model'''
  log_dict = {'epoch': log_data['epoch'], 'time_train': log_data['time_train']}
  log_dict[f'train/{config.loss.name}'] = log_data['train_loss']
  log_dict[f'val/{config.loss.name}']   = log_data['val_loss']

  for k,v in log_data['train_loss_logs'].items(): log_dict[f'train/{k}'] = v
  for k,v in log_data['val_loss_logs'].items():   log_dict[f'val/{k}']   = v

  ts_dict = sampler.get_timesteps()
  if sampler.st_train:
    log_dict.update({f'params/solver_timesteps_{n}': t.item() for n, t in enumerate(ts_dict['solver_timesteps'])})
  if sampler.dt_train:
    log_dict.update({f'params/diffusion_timesteps_{n}': t.item() for n, t in enumerate(ts_dict['diffusion_timesteps'])})
  if hasattr(sampler, "train_params"):
    log_dict['params/solver_absmax'] = max([i.abs().max().item() for i in sampler.train_params])
    solver_params = [p.detach().cpu().numpy().tolist() for p in sampler.train_params]
    log_list(name=f"solver_{log_data['global_step']}", data=solver_params, artifact_dir="artifacts/solver")
  
  log_scalars(log_dict, log_data['global_step'])

  if log_data['val_imgs_log']:
    train_imgs_teacher, train_imgs_log = list(zip(*log_data['train_imgs_log'].values()))
    val_imgs_teacher, val_imgs_log = list(zip(*log_data['val_imgs_log'].values()))

    train_imgs_teacher = train_imgs_teacher if train_imgs_teacher[0] else None
    val_imgs_teacher   = val_imgs_teacher   if val_imgs_teacher[0]   else None
    
    if train_imgs_log:
      mlflow_log_imgs(
        imgs_student=train_imgs_log,
        imgs_teacher=train_imgs_teacher,
        name=f"train_{log_data['epoch']}",
    )
    mlflow_log_imgs(
      imgs_student=val_imgs_log,
      imgs_teacher=val_imgs_teacher,
      name=f"val_{log_data['epoch']}",
    )

# ==================================================================================================================
# MLFLOW IMG LOG
# ==================================================================================================================
def mlflow_log_imgs(imgs_student, imgs_teacher=None, name="", downscale=256):
  """
  Build a side-by-side Student/Teacher grid figure and log it to MLflow.
  Mirrors your wandb_log_imgs behavior.
  """
  if isinstance(imgs_student, list):
    imgs_student = torch.stack(imgs_student, dim=0)
  if isinstance(imgs_teacher, list) and imgs_teacher is not None:
    imgs_teacher = torch.stack(imgs_teacher, dim=0)

  # Downscale
  if downscale is not None and imgs_student.shape[-1] > downscale:
    imgs_student = torch.nn.functional.interpolate(
      imgs_student, size=(downscale, downscale), mode='bilinear', align_corners=False
    ).squeeze()
    if imgs_teacher is not None:
      imgs_teacher = torch.nn.functional.interpolate(
        imgs_teacher, size=(downscale, downscale), mode='bilinear', align_corners=False
      ).squeeze()
      
  imgs_student = imgs_student.to(torch.float32)
  if imgs_teacher is not None:
    imgs_teacher = imgs_teacher.to(torch.float32)


  n_cols = 2 if imgs_teacher is not None else 1
  fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
  axes = ax if isinstance(ax, (list, np.ndarray)) else (ax if n_cols == 1 else [ax])

  # Student
  vis_grid(imgs_student, ax=axes[0] if n_cols == 2 else axes)
  (axes[0] if n_cols == 2 else axes).axis('off')
  (axes[0] if n_cols == 2 else axes).set_title("Student")

  # Teacher
  if imgs_teacher is not None:
    vis_grid(imgs_teacher, ax=axes[1])
    axes[1].axis('off')
    axes[1].set_title("Teacher")

  fig.tight_layout()
  mlflow.log_figure(fig, f"images/{name}.png")
  plt.close(fig)


def vis_grid(imgs_row, ax=None):
  imgs_row = imgs_row.detach().cpu()
  nrow = int(np.around(np.sqrt(imgs_row.shape[0])))
  imgs_grid = make_grid(imgs_row, nrow=nrow).permute(1, 2, 0).numpy()

  if imgs_grid.max() > 2:
    imgs_grid = imgs_grid / 255.0

  if imgs_grid.min() < -0.1:
    imgs_grid = imgs_grid / 2 + 0.5

  imgs_grid = np.clip(imgs_grid, 0, 1)
  if ax is None:
    plt.imshow(imgs_grid)
  else:
    ax.imshow(imgs_grid)


# -----------------------------
# Utilities
# -----------------------------
def _to_num(x):
  if isinstance(x, (int, float, bool)):
    return float(x)
  if torch.is_tensor(x) and x.numel() == 1:
    return float(x.detach().cpu().item())
  return None

def flatten_config(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
  """Flatten nested dicts; serialize lists/dicts as JSON strings for log_params()."""
  items = []
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
    if isinstance(v, dict):
      items.extend(flatten_config(v, new_key, sep=sep).items())
    elif isinstance(v, (list, tuple)):
      items.append((new_key, json.dumps(v, ensure_ascii=False)))
    else:
      items.append((new_key, v))
  return dict(items)

def log_config(config: Dict[str, Any], save_path: str = "config/config.json", log_params = False) -> None:
  """Log flattened params + the exact JSON config as an artifact."""
  flat = flatten_config(config)
  mlflow.log_params(flat)
  with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "config.json"
    p.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    mlflow.log_artifact(str(p), artifact_path=str(Path(save_path).parent))


def _end(status: str):
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except Exception:
        pass

def install_mlflow_guards():
    # Mark runs KILLED on common termination signals
    def _on_signal(signum, frame):
        _end("KILLED")
        os._exit(128 + signum)  # immediate exit, no extra cleanup races

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    # Mark FAILED on uncaught exceptions
    _orig = sys.excepthook
    def _hook(tp, exc, tb):
        _end("FAILED")
        _orig(tp, exc, tb)
    sys.excepthook = _hook

    # Last resort: if process exits normally but run still open → KILLED
    atexit.register(lambda: _end("KILLED"))