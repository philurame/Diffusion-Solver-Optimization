import torch, tqdm, os
import torch.utils.checkpoint as cp
from diffusers import StableDiffusionPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD15_CACHE = os.path.join(ROOT, 'data', 'cache')

class SD15(StableDiffusionPipeline):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    is_train = kwargs.get('is_train', False)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    local_files_only = kwargs.get('local_files_only', True)
    pipe = super().from_pretrained(
      "sd-legacy/stable-diffusion-v1-5", 
      cache_dir = SD15_CACHE,
      torch_dtype=torch.float32 if is_train else torch.float16,
      local_files_only=local_files_only
    ).to(device)

    pipe.is_train = is_train

    # pipe.config = {
    #   "num_train_timesteps": 1000,
    #   "beta_start": 0.00085,
    #   "beta_end": 0.012,
    #   "max_timestep": 999.5,
    # }
    
    for param in pipe.unet.parameters():
      param.requires_grad = False
    for param in pipe.vae.parameters():
      param.requires_grad = False
    return pipe
  

  def __call__(self, *args, **kwargs):
    if self.is_train:
      return self._call_impl(*args, **kwargs) 
    with torch.no_grad():
      return self._call_impl(*args, **kwargs)
  
  def _call_impl(self,
    prompt: str,
    num_inference_steps: int = 50,
    output_type="latent",
    height = 512, 
    width = 512,
    generator = None,
    **kwargs
    ):
    device = kwargs.get('device') or self._execution_device
    guidance_scale = kwargs.get('guidance_scale', 7.5)


    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
      prompt,
      device,
      num_images_per_prompt=1,
      do_classifier_free_guidance=guidance_scale>0,
      negative_prompt=kwargs.get("negative_prompt", None),
      prompt_embeds=kwargs.get("prompt_embeds", None),
      negative_prompt_embeds=kwargs.get("negative_prompt_embeds", None),
      lora_scale=None,
      clip_skip=None,
    )

    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    input_noise = kwargs.get("noise", None)
    if input_noise is None:
      input_noise = kwargs.get("latents", None)
      
    latents = self.prepare_latents(
      batch_size = batch_size,
      num_channels_latents=self.unet.config.in_channels,
      height=height,
      width=width,
      device=device,
      dtype=prompt_embeds.dtype,
      generator=generator,
      latents=input_noise,
    ) 
    self.input_noise = latents

    if guidance_scale>0:
      prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
      self.prompt_embeds = prompt_embeds


    diffusion_timesteps = self.set_sampler(num_inference_steps, latents, device, **kwargs)

    if kwargs.get('dry_run', False):
      # need to set sampler for optimizer
      return None

    # Diffusion steps
    for t in tqdm.tqdm(diffusion_timesteps, disable=not kwargs.get('verbose', False)):

      if self.is_train:
        noise_pred = self.make_diffusion_step_cp(latents, t, guidance_scale)
      else:
        noise_pred = self.make_diffusion_step(latents, t, guidance_scale)

      latents = self.sampler.step(model_output=noise_pred, sample=latents)
      if not isinstance(latents, torch.Tensor): latents = latents[0]
    
    # return latents
    if output_type == "latent":
      return latents

    # return pt image
    imgs_pt = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
    imgs_pt = self.image_processor.postprocess(imgs_pt, output_type='pt')
    if output_type == "pt": # returns -1 -> 1
      return imgs_pt * 2 - 1
    
    # return 255 image
    imgs_255 = (imgs_pt*255).clip(0,255).to(device='cpu', dtype=torch.uint8)
    return imgs_255
  

  def make_diffusion_step(self, latents, t, guidance_scale):
    do_cfg = guidance_scale>0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

    noise_pred = self.unet(
      latent_model_input,
      t,
      timestep_cond=None,
      cross_attention_kwargs=None,
      return_dict=False,
      encoder_hidden_states=self.prompt_embeds,
      added_cond_kwargs={},
    )[0]    

    if do_cfg:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred
  
  def make_diffusion_step_cp(self, latents, t, guidance_scale):
    do_cfg = guidance_scale>0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

    noise_pred = cp.checkpoint(
      self.unet,
      latent_model_input,
      t,
      use_reentrant = False,
      timestep_cond=None,
      cross_attention_kwargs=None,
      return_dict=False,
      encoder_hidden_states=self.prompt_embeds,
      added_cond_kwargs={},
    )[0]

    if do_cfg:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


  def set_sampler(self, num_inference_steps=None, latents=None, device=None, **kwargs):
    solver_timesteps = kwargs.get('solver_timesteps', None)
    if solver_timesteps is None:
      solver_timesteps = kwargs.get('timesteps', None)
      
    device=device or self.device

    # set solver_timesteps and diffusion_timesteps
    self.sampler.set_timesteps(
      num_inference_steps=num_inference_steps, 
      solver_timesteps=solver_timesteps,
      device=device,
      **kwargs
    )

    # set solver train params
    if hasattr(self.sampler, "set_train_solver"):
      self.sampler.set_train_solver(latents, **kwargs)

    diffusion_timesteps = self.sampler.get_timesteps(**kwargs)['diffusion_timesteps']
    if kwargs.get('diffusion_timesteps', None) is not None:
      diffusion_timesteps = kwargs['diffusion_timesteps']
    
    return diffusion_timesteps