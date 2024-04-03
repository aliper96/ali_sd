import os

import clip
import numpy as np
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_folder_features
from tqdm import tqdm
import wandb
wandb.login(key="7932525c36d2cd0d2995858bb0a4f432e982e0b8",relogin=True)
from ali2ali_text import Ali2Ali
from utils import parse_args_paired_training, PairedDataset
import torch
import lpips
import vision_aided_loss
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


def main(args):
    wandb.init(project=args.tracker_project_name, entity="aliper96")

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    # create model pix 2 pix
    net_pix2pix = Ali2Ali(lora_rank_unet=8, lora_rank_vae=4)
    net_pix2pix.set_train()
    # create model discriminator
    net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps ,
        num_training_steps=args.max_train_steps ,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps ,
            num_training_steps=args.max_train_steps ,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    # dataset_val = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)
    # dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",)

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False



    global_step = 0
    for epoch in range(0,args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            # x_src = batch["conditioning_pixel_values"].cuda()
            x_tgt = batch["output_pixel_values"].cuda()
            input_ids = batch["input_ids"].cuda()
            B,C,H,W = x_tgt.shape
            #forward pass
            x_tgt_pred = net_pix2pix(x_tgt,prompt_tokens = input_ids ,deterministic = True)
            loss_l2 = F.mse_loss(x_tgt.float(), x_tgt_pred.float()).mean() * args.lambda_l2
            loss_lpips = F.mse_loss(x_tgt.float(), x_tgt_pred.float()).mean() * args.lambda_lpips
            loss = loss_l2 + loss_lpips
            # CLIP similarity loss
            if args.lambda_clipsim > 0:
                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                loss_clipsim = (1 - clipsim.mean() / 100)
                loss += loss_clipsim * args.lambda_clipsim


            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Generator loss: fool the discriminator

            x_tgt_pred = net_pix2pix(x_tgt, prompt_tokens=input_ids, deterministic=True)
            lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Discriminator loss: fake image vs real image
            lossD_real = (net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan).mean()
            lossD_real.backward()
            torch.nn.utils.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
            optimizer_disc.step()
            lr_scheduler_disc.step()
            optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
            # fake image
            lossD_fake = (net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan).mean()
            lossD_fake.backward()
            torch.nn.utils.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
            optimizer_disc.step()
            optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
            lossD = lossD_real + lossD_fake


            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            global_step += 1

            logs = {}
            # log all the losses
            logs["lossG"] = lossG.detach().item()
            logs["lossD"] = lossD.detach().item()
            logs["loss_l2"] = loss_l2.detach().item()
            logs["loss_lpips"] = loss_lpips.detach().item()
            if args.lambda_clipsim > 0:
                logs["loss_clipsim"] = loss_clipsim.detach().item()
            progress_bar.set_postfix(**logs)

            decoded_prompts = []
            for ids in input_ids:
                ids_squeezed = ids.squeeze(0)  # Elimina la primera dimensión si su tamaño es 1
                decoded_prompt = net_pix2pix.tokenizer.decode(ids_squeezed.tolist(), skip_special_tokens=True)
                decoded_prompts.append(decoded_prompt)
            # viz some images
            if global_step % args.viz_freq == 1:
                log_dict = {
                    # "train/decoded_prompts": [wandb.Html(f"<p>{prompt}</p>") for prompt in decoded_prompts],
                    "train/decoded_prompts": [wandb.Html(f"<p>{decoded_prompts[idx]}</p>") for idx in range(B)],
                    "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                    "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                }
                for k in log_dict:
                    logs[k] = log_dict[k]
            wandb.log(logs, step=global_step)
            # checkpoint the model
            if global_step % args.checkpointing_steps == 1:
                outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                net_pix2pix.save_model(outf)





if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)

