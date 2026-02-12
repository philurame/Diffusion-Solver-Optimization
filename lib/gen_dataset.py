from .model import construct_pipeline
from tqdm import tqdm
import torch, sys, os, tqdm, pickle

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PROMPTS = os.path.join(DATA, 'coco_10k.txt')

device    = f'cuda:{sys.argv[1]}'
seed_from = int(sys.argv[2])
seed_to   = int(sys.argv[3])

print(device, seed_from, seed_to)

with open(PROMPTS, 'r') as f:
  prompts = [i.strip() for i in f.readlines()]
  N = len(prompts[seed_from:seed_to])


pipe = construct_pipeline(theor_solver='DDIM', device=device, is_train=False, local_files_only=True)

teacher = torch.zeros(N, 3, 512, 512, dtype=torch.uint8)
noise   = torch.zeros(N, 4, 64, 64,   dtype=torch.float16)

batch = 64
for s in tqdm.tqdm(range(seed_from, seed_to, batch)):
  generator = [torch.Generator('cpu').manual_seed(i) for i in range(s, s+batch)]
  imgs = pipe(
    prompt = prompts[s:s+batch],
    generator = generator,
    num_inference_steps=100,
    output_type='img',
  )
  teacher[s:s+batch] = imgs
  noise[s:s+batch] = pipe.input_noise.clone()


with open(os.path.join(DATA, f'dataset_{seed_from}_{seed_to}.pkl'), 'wb') as f:
  pickle.dump({
    'prompts': prompts[seed_from:seed_to],
    'imgs':  teacher,
    'noise': noise,
  }, f)

