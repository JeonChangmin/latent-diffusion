from argparse import Namespace
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
import yaml

from ldm.models.autoencoder import AutoencoderKL  # First Stage
from ldm.modules.encoders.modules import BERTEmbedder  # Condition Model
from ldm.modules.diffusionmodules.openaimodel import UNetModel  # Diffusion Model


class LatentDiffusion(nn.Module):
    def __init__(
        self,
        unet_config,
        first_stage_config,
        cond_stage_config,
        scale_factor=1.0,
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
    ):
        super().__init__()
        self.model = UNetModel(**unet_config)
        self.first_stage_model = AutoencoderKL(**first_stage_config)
        self.cond_stage_model = BERTEmbedder(**cond_stage_config)
        self.scale_factor = scale_factor
        self.num_timesteps = int(timesteps)

        betas = (
            np.linspace(linear_start**0.5, linear_end**0.5, timesteps, dtype=np.float64)
            ** 2
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        assert timesteps == betas.shape[0] == alphas.shape[0] == alphas_cumprod.shape[0]

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(sqrt_one_minus_alphas_cumprod)
        )

    def forward(self, x, t, cond):
        return self.model(x, t, context=cond)


class DDIMSamper:
    def __init__(self, model: LatentDiffusion):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        ddim_steps,
        batch_size,
        img_shape,
        conditioning=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        temperature=1.0,
    ):
        ddim_timesteps = (
            np.arange(0, self.ddpm_num_timesteps, self.ddpm_num_timesteps // ddim_steps)
            + 1
        )

        device = self.model.alphas_cumprod.device
        alphas_cumprod = self.model.alphas_cumprod.cpu()
        ddim_alphas = alphas_cumprod[ddim_timesteps]
        ddim_alphas_prev = np.hstack([alphas_cumprod[0], alphas_cumprod[ddim_timesteps[:-1]]])
        ddim_sqrt_one_minus_alphas = np.sqrt(1.0 - ddim_alphas)
        ddim_sigmas = np.zeros(len(ddim_timesteps), dtype=np.float32)

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32).to(device)
        self.ddim_alphas = to_torch(ddim_alphas)
        self.ddim_alphas_prev = to_torch(ddim_alphas_prev)
        self.ddim_sqrt_one_minus_alphas = to_torch(ddim_sqrt_one_minus_alphas)
        self.ddim_sigmas = to_torch(ddim_sigmas)

        samples = torch.randn((batch_size, *img_shape), device=device)  # 4, 4, 32, 32
        total_steps = len(ddim_timesteps)
        iterator = tqdm(
            np.flip(ddim_timesteps),
            total=total_steps,
        )
        for i, step in enumerate(iterator):
            timestamp = torch.full((batch_size,), step, device=device, dtype=torch.long)  # 981s for the first loop
            index = total_steps - i - 1  # 49 for the first loop
            samples = self.sample_ddim(
                samples,
                conditioning,
                timestamp,
                index=index,
                temperature=temperature,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return samples

    @torch.no_grad()
    def sample_ddim(
        self,
        x,
        c,  # -0.0047, -0.0047
        t,
        index,
        temperature=1.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        device = x.device
        b = x.shape[0]

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, c])
        e_t_uncond, e_t = self.model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn(x.shape, device=device) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev


def state_dict_of(state_dict_path: str) -> Dict[str, Any]:
    state_dict = torch.load(state_dict_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            new_k = k.replace("model.diffusion_model.", "model.")
            state_dict[new_k] = state_dict.pop(k)
    return state_dict


if __name__ == "__main__":
    opt = Namespace()
    opt.prompt = "a virus monster is playing guitar, oil on canvas"
    opt.n_samples = 4
    opt.ddim_steps = 50
    opt.scale = 5.0
    opt.H = 256
    opt.W = 256
    opt.outdir = "outputs/txt2img-samples"
    print(opt)

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    with Path("stable-diffusion.yaml").open("r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        pprint(model_config)
    model = LatentDiffusion(**model_config)
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict_of("/data1/diffusion/stable-diffusion.ckpt"), strict=False
    )
    assert not missing_keys
    model.eval()
    model.to(device)

    sampler = DDIMSamper(model)

    sample_path = Path(opt.outdir) / "samples"
    sample_path.mkdir(parents=True, exist_ok=True)
    base_count = len(list(sample_path.glob("*.png")))

    with torch.no_grad():
        uc = model.cond_stage_model.encode(opt.n_samples * [""])
        c = model.cond_stage_model.encode(opt.n_samples * [opt.prompt])
        img_shape = [4, opt.H // 8, opt.W // 8]
        samples_ddim = sampler.sample(
            ddim_steps=opt.ddim_steps,
            batch_size=opt.n_samples,
            img_shape=img_shape,
            conditioning=c,
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
        )
        samples_ddim = 1.0 / model.scale_factor * samples_ddim
        x_samples_ddim = model.first_stage_model.decode(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            Image.fromarray(x_sample.astype(np.uint8)).save(
                sample_path / f"{base_count:04}.png"
            )
            base_count += 1
