import os
import torch
import signal
import multiprocessing as mp
from queue import Empty

from lib.model import construct_pipeline

# ========================================================================
# KILL all processes when parent dies
# ========================================================================
def _install_parent_deathsig():
  try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
  except Exception: 
    pass

def _install_signal_handlers():
  def _exit_fast(signum, frame):
    try: torch.cuda.empty_cache()
    except Exception: pass
    os._exit(0)
  for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try: signal.signal(sig, _exit_fast)
    except Exception: pass

# ========================================================================
# Main mp loop
# ========================================================================
def _worker_loop(device_id, shared_cfg, in_q, out_q):
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  _install_parent_deathsig()
  _install_signal_handlers()

  torch.cuda.set_device(device_id)
  device_str = f"cuda:{device_id}"
  parent_pid_at_start = os.getppid()

  pipe = construct_pipeline(
    theor_solver=shared_cfg["theor_solver"], 
    device=device_str, 
    train_solver=shared_cfg.get("train_solver", None), 
    device=device_str,
    is_train=False,
  )

  while True:
    cur_ppid = os.getppid()
    if cur_ppid == 1 or cur_ppid != parent_pid_at_start:
      break

    try:
      task = in_q.get(timeout=1.0)
    except Empty: 
      continue

    if task is None:
      break

    job_id = task["job_id"]
    prompts = task["prompts"]
    solver_timesteps = task.get("solver_timesteps", None)   
    diffusion_timesteps = task.get("diffusion_timesteps", None)   
    solver_params = task.get("solver_params", None)
    noise = task.get("noise", None)
    pipe_kwargs = task.get("pipe_kwargs", {})
    if noise is not None:
      noise = noise.to(device_str, dtype=torch.float16)
    
    if hasattr(pipe.sampler, "train_params") and solver_params is not None:
      torch.nn.utils.vector_to_parameters(solver_params, pipe.sampler.train_params)

    with torch.no_grad():
      student = pipe(
        prompt=prompts,
        solver_timesteps=solver_timesteps,
        diffusion_timesteps=diffusion_timesteps,
        noise=noise,
        output_type="img",
        **pipe_kwargs
      )
    out_q.put((job_id, student.cpu()))
    torch.cuda.empty_cache()

def start_pool(devices, shared_cfg):
  ctx = mp.get_context("spawn")
  in_queues, procs = [], []
  out_q = ctx.Queue()

  for device in [int(d) for d in devices]:
    in_q = ctx.Queue()
    p = ctx.Process(
      target=_worker_loop,
      args=(device, shared_cfg, in_q, out_q),
      daemon=False
    )
    p.start()
    in_queues.append(in_q)
    procs.append(p)

  return {"procs": procs, "in_queues": in_queues, "out_q": out_q, "devices": [int(d) for d in devices]}

def stop_pool(pool):
  for q in pool["in_queues"]:
    try: q.put(None)
    except Exception: pass
  for p in pool["procs"]:
    try: p.join(timeout=10)
    except Exception: pass

def map_jobs(pool, jobs):
  in_queues = pool["in_queues"]
  out_q = pool["out_q"]
  n_workers = len(in_queues)

  for i, job in enumerate(jobs):
    in_queues[i % n_workers].put(job)

  results = {}
  for _ in range(len(jobs)):
    job_id, payload = out_q.get()
    results[job_id] = payload

  return [results[j["job_id"]] for j in jobs]
