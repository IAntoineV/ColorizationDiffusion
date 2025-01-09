import torch
from diffusion_conditioned import DiffusionConditioned
from channel_engineering import normalize_lab, rgb_to_lab, lab_to_rgb, unnormalize_lab

class ColorizationModule:
    def __init__(self, diffusion:DiffusionConditioned, transform=None):
        self.diffusion = diffusion
        self.transform = transform

    def __call__(self, img, T=None, to_log=False):
        infos={}
        img_processed=img.to(self.diffusion.device)
        if self.transform is not None:
            img_processed = self.transform(img_processed).to(self.diffusion.device)
        if len(img_processed.shape) != 4:
         img_processed = img_processed.unsqueeze(0)
        lab_images = normalize_lab(rgb_to_lab(img_processed, device=self.diffusion.device))
        imgs_l,imgs_ab = lab_images[:,0:1], lab_images[:,1:]
        shape = imgs_ab.shape
        if T is None:
            img_gen,logs = self.diffusion.sampling(imgs_l, shape, to_log=to_log)
            if to_log:
                infos["logs"] = logs
        else:
            noised = self.diffusion.forward_diffusion(imgs_ab, T)[0]
            img_noised_lab = torch.concat([imgs_l, noised], dim=1)
            img_noised_rgb = lab_to_rgb(img_noised_lab, device=img_noised_lab.device)
            infos["img_noised"] = img_noised_rgb.cpu()
            img_gen,logs = self.diffusion.sampling(imgs_l, shape, T=T, xT=noised, to_log=to_log)
            if to_log:
                infos["logs"] = logs

        img_lab = torch.concat([imgs_l, img_gen], dim=1)
        img_rgb = lab_to_rgb(unnormalize_lab(img_lab), device=img_lab.device)
        if to_log:
            for t,img in infos["logs"].items():
                infos["logs"][t] = lab_to_rgb(unnormalize_lab(torch.concat([imgs_l, img], dim=1)), device=img_lab.device).cpu()
        return img_rgb.cpu(), infos
