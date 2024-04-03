import torch
from diffusers import AutoencoderKL
from lpips import lpips
import torch.nn.functional as F
from transformers import AutoTokenizer
import wandb
wandb.login(key="7932525c36d2cd0d2995858bb0a4f432e982e0b8",relogin=True)
from utils import PairedDataset, parse_args_paired_training


def main(args):
    wandb.init(project="sd_vae", entity="aliper96")

    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae = AutoencoderKL(in_channels=3,
                        out_channels=3,
                        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",
                                          "DownEncoderBlock2D"),
                        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",
                                        "UpDecoderBlock2D"),
                        block_out_channels=(32, 32, 32, 32),
                        layers_per_block=2,
                        norm_num_groups=32,
                        act_fn="silu",
                        latent_channels=4,
                        sample_size=768)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    vae.to("cuda")


    #train the vae
    vae.train()
    # vae.encoder.requires_grad_(True)
    # vae.decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(list(vae.encoder.parameters()) + list(vae.decoder.parameters()), lr=1e-4)
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")

    dataset_train = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", tokenizer=tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size= 1, shuffle=True, num_workers=args.dataloader_num_workers)
    global_step = 0
    for i in range(1000):
        for step, batch in enumerate(dl_train):
            x_tgt = batch["output_pixel_values"].cuda()
            B,C,H,W = x_tgt.shape


            encoded_control = vae.encode(x_tgt).latent_dist.sample() * vae.config.scaling_factor
            output_image = (vae.decode(encoded_control / vae.config.scaling_factor).sample).clamp(-1, 1)
            loss_l2 = F.mse_loss(x_tgt.float(), output_image.float(), reduction="mean") * 1.0 #args.lambda_l2
            loss_lpips = net_lpips(x_tgt.float(), output_image.float()).mean() * 5 # args.lambda_lpips
            loss = loss_l2 + loss_lpips
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            logs = {}
            # log all the losses
            logs["loss_l2"] = loss_l2.detach().item()
            logs["loss_lpips"] = loss_lpips.detach().item()

            # viz some images
            if global_step % 50 == 1:
                log_dict = {
                    "train/source": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in
                                     range(B)],

                    "train/model_output": [wandb.Image(output_image[idx].float().detach().cpu(), caption=f"idx={idx}") for
                                           idx in range(B)],
                }
                for k in log_dict:
                    logs[k] = log_dict[k]
            wandb.log(logs, step=global_step)



if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)