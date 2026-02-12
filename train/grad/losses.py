import torch, sys, os
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
LIB = os.path.join(ROOT, 'lib')
sys.path.insert(0, LIB)
from lib.registries import ClassRegistry
loss_registry = ClassRegistry()


# ========================================================================================
# L1
# ========================================================================================
@loss_registry.add_to_registry("L1")
class L1(torch.nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

  def forward(self, student=None, teacher=None, **kwargs):
    loss = F.l1_loss(student, teacher, reduction='none')
    loss = loss.mean(dim=list(range(1,len(loss.shape))))
    return loss


# ========================================================================================
# LPIPS
# ========================================================================================  
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

@loss_registry.add_to_registry("LPIPS")
class LPIPS(torch.nn.Module):
  net = "alex"
  def __init__(self, device, **kwargs):
    super().__init__()
    self.device = device

    self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=self.net, reduction="none", normalize=False).to(device)
    self.lpips.eval()
    for p in self.lpips.parameters():
      p.requires_grad_(False)

  def forward(self, student=None, teacher=None, **kwargs):
    student = self.normalize(student)
    teacher = self.normalize(teacher)
    lpips_scores = self.lpips(student.to(self.device), teacher.to(self.device))
    return lpips_scores  # [B]

  def normalize(self, imgs):
    """[0,1], uint8 -> [-1,1]"""
    if imgs.ndim >= 4 and imgs.shape[-1] == 3:
        imgs = imgs.movedim(-1, -3)
    if imgs.dtype.is_floating_point and imgs.min() > -0.001:
        imgs = imgs * 2 - 1
    if imgs.dtype == torch.uint8:
        imgs = imgs.float().div(127.5) - 1
    return imgs.clamp(-1, 1)


# ========================================================================================
# HPS
# ========================================================================================
from torchvision.transforms.functional import resize as tv_resize, center_crop as tv_center_crop
from torchvision.transforms.functional import InterpolationMode
from hpsv2.img_score import initialize_model, model_dict
from hpsv2.utils import hps_version_map
from huggingface_hub import hf_hub_download
from hpsv2.src.open_clip import get_tokenizer

@loss_registry.add_to_registry("HPS")
class HPS(torch.nn.Module):
  def __init__(self, device, **kwargs):
    super().__init__()
    self.device = device

    self.tokenizer = get_tokenizer("ViT-H-14")

    torch.cuda.set_device(self.device) # TODO: check on leakage
    initialize_model()
    
    self.model = model_dict["model"]
    ckpt_path = hf_hub_download("xswu/HPSv2", hps_version_map['v2.1'])
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    self.model.load_state_dict(state, strict=True)
    self.model.to(device)

    self.model.eval()
    for param in self.model.parameters():
      param.requires_grad = False
    
  def forward(self, student=None, prompts=None, **kwargs):
    tokens = self.tokenizer(prompts).to(self.device, non_blocking=True)  # [N, context_len]
    student = self.normalize(student).to(self.device, non_blocking=True)
    feats = self.model(student, tokens)
    img_f = feats["image_features"] # [B, D], L2-normalized
    txt_f = feats["text_features"]  # [B, D], L2-normalized
    return -(img_f * txt_f).sum(dim=1) # [B]

  def normalize(self, imgs, image_size=224):
    '''[-1,1], uint8 -> [0,1] -> crop + N(0,1)'''
    if imgs.ndim >= 4 and imgs.shape[-1] == 3:
      imgs = imgs.movedim(-1, -3)
    if imgs.dtype.is_floating_point and imgs.min() < -0.001:
      imgs = imgs / 2 + 0.5        
    if imgs.dtype == torch.uint8:
      imgs = imgs.float().div(255)
    imgs = imgs.clamp(0,1)

    imgs = tv_resize(imgs, size=image_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    imgs = tv_center_crop(imgs, output_size=[image_size, image_size])
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=imgs.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=imgs.device).view(1, 3, 1, 1)
    return (imgs - mean) / std


# ========================================================================================
# HPS + LPIPS
# ========================================================================================
@loss_registry.add_to_registry("HPS-LPIPS")
class HPS_LPIPS(torch.nn.Module):
  def __init__(self, device, hps_coef, **kwargs):
    super().__init__()
    self.device = device
    self.lpips = LPIPS(device, **kwargs)
    self.hps   = HPS(device, **kwargs)
    self.hps_coef = hps_coef
    self.logs = {'lpips': [0.,0], 'hps': [0.,0]}

  def forward(self, student=None, teacher=None, prompts=None, **kwargs):
    lp = self.lpips(student=student, teacher=teacher, **kwargs).to(self.device)
    hp = self.hps(student=student, prompts=prompts, **kwargs).to(self.device)

    for m, m_name in [[lp, 'lpips'], [hp, 'hps']]:
      m_sum = m.detach().sum().item()
      avg, cnt = self.logs[m_name]
      self.logs[m_name][0] = (avg * cnt + m_sum) / (cnt + m.numel())
      self.logs[m_name][1] = cnt + m.numel()

    return lp + hp * self.hps_coef