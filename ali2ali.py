from diffusers import AutoencoderKL,UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel
import torch
from utils import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class Ali2Ali(torch.nn.Module):
    def __init__(self,lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # vae = AutoencoderKL(in_channels=3,
        #                          out_channels=3,
        #                          down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",
        #                                            "DownEncoderBlock2D"),
        #                          up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",
        #                                          "UpDecoderBlock2D"),
        #                          block_out_channels=(128, 256, 512, 512),
        #                          layers_per_block=2,
        #                         norm_num_groups=32,
        #                          act_fn="silu",
        #                          latent_channels=4,
        #                          sample_size=768)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # self.unet = UNet2DConditionModel(sample_size=64,
        #                                  in_channels=4,
        #                                  out_channels=4,
        #                                  center_input_sample=False,
        #                                  flip_sin_to_cos=True,
        #                                  freq_shift=0,
        #                                  down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        #                                                    "CrossAttnDownBlock2D", "DownBlock2D"),
        #                                  up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
        #                                                  "CrossAttnUpBlock2D"),
        #                                  block_out_channels=(320, 640, 1280, 1280),
        #                                  layers_per_block=2,
        #                                  downsample_padding=1,
        #                                  mid_block_scale_factor=1,
        #                                  act_fn="silu",
        #                                  norm_num_groups=32,
        #                                  norm_eps=1e-5,
        #                                  cross_attention_dim=768,
        #                                  attention_head_dim=8)

        #skip connections
        print("Initializing model with random weights")
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                              "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                              "to_k", "to_q", "to_v", "to_out.0",
                              ]
        vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                                     target_modules=target_modules_vae)
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                                      target_modules=target_modules_unet
                                      )
        unet.add_adapter(unet_lora_config)
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.conv_in.requires_grad_(True)
        self.unet.down_blocks.requires_grad_(True)
        self.unet.mid_block.requires_grad_(True)
        self.unet.up_blocks.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)




if __name__ == "__main__":
    Ali2Ali(lora_rank_unet=8, lora_rank_vae=4)