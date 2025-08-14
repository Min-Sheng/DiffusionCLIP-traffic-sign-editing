import os
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from improved_ddpm.script_util import i_DDPM
from utils import dict2namespace, get_beta_schedule, denoising_step

PRETRAINED_MODEL_PATH = 'pretrained/512x512_diffusion.pt'


class DiffusionCLIPInferencer:
    def __init__(
        self,
        edit_type='snowy',
        degree_of_change=1.0,
        t_0=301,
        fine_tuned_epoch=9,
        config_file='traffic_sign.yml',
        device=None,
    ):
        """
        Initializes the inferencer with all necessary configurations.
        """
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.t_0 = t_0
        self.degree_of_change = degree_of_change
        self.edit_type = edit_type
        
        # --- Paths and Model Checkpoints ---
        fine_tuned_ckpt_dir = Path('./checkpoints')
        fine_tuned_ckpt_map = {
            'night': fine_tuned_ckpt_dir / f'traffic_sign_night_FT_traffic_sign_traffic_sign_night_t{t_0}_ninv40_ngen6_id0.0_l11.0_lr8e-06_traffic_sign_in_darkness-{fine_tuned_epoch}.pth',
            'rainy': fine_tuned_ckpt_dir / f'traffic_sign_rainy_FT_traffic_sign_traffic_sign_rainy_t{t_0}_ninv40_ngen6_id0.0_l11.0_lr8e-06_traffic_sign_during_rainy_weather-{fine_tuned_epoch}.pth',
            'foggy': fine_tuned_ckpt_dir / f'traffic_sign_foggy_FT_traffic_sign_traffic_sign_foggy_t{t_0}_ninv40_ngen6_id0.0_l11.0_lr8e-06_traffic_sign_in_dense_fog_-{fine_tuned_epoch}.pth',
            'snowy': fine_tuned_ckpt_dir / f'traffic_sign_snowy_FT_traffic_sign_traffic_sign_snowy_t{t_0}_ninv40_ngen6_id0.0_l11.0_lr8e-06_traffic_sign_in_snowy_weather-{fine_tuned_epoch}.pth',
            'rusty': fine_tuned_ckpt_dir / f'traffic_sign_rusty_FT_traffic_sign_traffic_sign_rusty_t{t_0}_ninv40_ngen6_id0.0_l11.0_lr8e-06_rusty_traffic_sign_with_faded_paint-{fine_tuned_epoch}.pth',
            'vines': fine_tuned_ckpt_dir / f'traffic_sign_vines_FT_traffic_sign_traffic_sign_vines_t{t_0}_ninv40_ngen6_id0.0_l11.0_lr8e-06_traffic_sign_overgrown_with_vines-{fine_tuned_epoch}.pth',
        }
        self.fine_tuned_ckpt = str(fine_tuned_ckpt_map[edit_type])
        print(f"Using checkpoint: {self.fine_tuned_ckpt}")

        # --- Load Config ---
        config_path = os.path.join('configs', config_file)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config = dict2namespace(self.config)
        self.config.device = self.device

        # --- Setup Diffusion Parameters ---
        self.model_var_type = self.config.model.var_type
        betas = get_beta_schedule(
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.models = self._load_models()
        self.learn_sigma = True

    def _load_models(self):
        models = []
        model_paths = [PRETRAINED_MODEL_PATH, self.fine_tuned_ckpt] # [Base Model, Fine-tuned Model]

        for model_path in model_paths:
            model_i = i_DDPM(self.config.data.dataset)
            if model_path:
                ckpt = torch.load(model_path, map_location=self.device)
                model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i.eval()
            models.append(model_i)
        
        return models

    def inference(self, image, n_inv_step=40, n_test_step=12):
        """
        Runs the full inference pipeline on a single image.
        """
        # --- Create Arguments Namespace ---        
        args_dic = {
            't_0': self.t_0,
            'n_inv_step': int(n_inv_step),
            'n_test_step': int(n_test_step),
            'sample_type': 'ddim',
            'eta': 0.0,
            'bs_test': 1,
            'model_path': self.fine_tuned_ckpt,
            'deterministic_inv': True,
            'n_iter': 1,
            'model_ratio': self.degree_of_change,
        }
        self.args = dict2namespace(args_dic)

        # ----------- Data -----------#
        n = self.args.bs_test

        img = image.resize((self.config.data.image_size, self.config.data.image_size))
        
        img_np = np.array(img)/255.0
        img_torch = torch.from_numpy(img_np).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img_torch = img_torch.to(self.device)
        x0 = (img_torch - 0.5) * 2.
        with torch.no_grad():
            #---------------- Invert Image to Latent -------------------#
            if self.args.deterministic_inv:
                seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                seq_inv = [int(s) for s in list(seq_inv)]
                seq_inv_next = [-1] + list(seq_inv[:-1])

                x = x0.clone()
                with tqdm(total=len(seq_inv), desc=f"Inversion process") as progress_bar:
                    for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_prev = (torch.ones(n) * j).to(self.device)
                        x = denoising_step(x, t=t, t_next=t_prev, models=self.models,
                                            logvars=self.logvar, sampling_type='ddim',
                                            b=self.betas, eta=0, learn_sigma=self.learn_sigma, ratio=0)
                        progress_bar.update(1)
                    x_lat = x.clone()
            
            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
            else:
                seq_test = list(range(self.args.t_0))
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()

                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        x = denoising_step(x, t=t, t_next=t_next, models=self.models,
                                           logvars=self.logvar, sampling_type=self.args.sample_type,
                                           b=self.betas, eta=self.args.eta, learn_sigma=self.learn_sigma,
                                           ratio=self.args.model_ratio)
                        progress_bar.update(1)
        
        # ----------- Convert Final Tensor to PIL Image and Return -----------#
        final_tensor = (x[0].detach().cpu() + 1) * 0.5
        final_tensor = final_tensor.clamp(0, 1)
        final_numpy = (final_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        edited_image = Image.fromarray(final_numpy)
        
        return edited_image