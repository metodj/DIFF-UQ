import os
import tqdm
from collections import Counter

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from models.diffusion import Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from gen_unc_laplace import DiffusionLLDiagLaplace, preprocess_la_adm, LaplaceDataset
from utils import inverse_data_transform, singlestep_ddim_sample, parse_args_and_config, seed_everything, get_beta_schedule


def main(args, config):
    print('Running main_epistemic_uncertainty')

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    # set random seed
    seed_everything(args.seed)
    fixed_xT = torch.randn([args.total_n_sample, config.data.channels, config.data.image_size, config.data.image_size]) 
    total_n_samples = args.total_n_sample
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(total_n_samples, args.sample_batch_size))
    n_rounds = total_n_samples // args.sample_batch_size
    if args.fixed_class == 10000:
        fixed_classes = torch.randint(low=0, high=1000, size=(args.sample_batch_size, n_rounds))
    else:
        fixed_classes = torch.randint(low=args.fixed_class, high=args.fixed_class+1, size=(args.sample_batch_size,n_rounds)).to(device)


    ######  initialize diffusion and model(unet) ########## 
    if config.model.model_type == "guided_diffusion":
        model = GuidedDiffusion_Model(
            image_size=config.model.image_size,
            in_channels=config.model.in_channels,
            model_channels=config.model.model_channels,
            out_channels=config.model.out_channels,
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attention_resolutions,
            dropout=config.model.dropout,
            channel_mult=config.model.channel_mult,
            conv_resample=config.model.conv_resample,
            dims=config.model.dims,
            num_classes=config.model.num_classes,
            use_checkpoint=config.model.use_checkpoint,
            use_fp16=config.model.use_fp16,
            num_heads=config.model.num_heads,
            num_head_channels=config.model.num_head_channels,
            num_heads_upsample=config.model.num_heads_upsample,
            use_scale_shift_norm=config.model.use_scale_shift_norm,
            resblock_updown=config.model.resblock_updown,
            use_new_attention_order=config.model.use_new_attention_order,
        )
    
    else:
        model = Model(config)

    model = model.to(device)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.device}


    if "ckpt_dir" in config.model.__dict__.keys():
        ckpt_dir = os.path.expanduser(config.model.ckpt_dir)
        states = torch.load(
            ckpt_dir,
            map_location=map_location
        )
        # states = {f"module.{k}":v for k, v in states.items()}
        if config.model.model_type == 'improved_ddpm' or config.model.model_type == 'guided_diffusion':
            model.load_state_dict(states, strict=True)
            if config.model.use_fp16:
                model.convert_to_fp16()
        else:
            modified_states = {}
            for key, value in states[0].items():
                modified_key =  key[7:]
                modified_states[modified_key] = value
            model.load_state_dict(modified_states, strict=True)

        if config.model.ema: 
            raise NotImplementedError()
        else:
            ema_helper = None

    image_path = '/nvmestore/mjazbec/imagenet_data/data_train/data'
    la_dataset = LaplaceDataset(device, image_path, image_size=config.model.image_size, train_la_data_size=args.train_la_data_size)
    la_dataloader= torch.utils.data.DataLoader(la_dataset, batch_size=args.train_la_batch_size, shuffle=True)

    # # print model layers
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel()}")
    

    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float()
    num_timesteps = betas.shape[0]

    _preprocess_la_adm = lambda x, y, device: preprocess_la_adm(x, y, betas, betas.shape[0], device)
    la = DiffusionLLDiagLaplace(model, f_preprocess_la_input=_preprocess_la_adm, last_layer_name="out.2",)
    la.fit(la_dataloader)
    last_layers = la.sample(args.mc_size)

    # add MAP model
    last_layers = torch.concat([la.mean[None, :], last_layers], dim=0)

    print(last_layers.shape)
    print(type(la))

    if args.skip_type == "uniform":
        skip = num_timesteps // args.timesteps
        seq = range(0, num_timesteps, skip)
    elif args.skip_type == "quad":
        seq = (
            np.linspace(
                0, np.sqrt(num_timesteps * 0.8), args.timesteps
            )
            ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError 

    EXP_ROOT = "/nvmestore/mjazbec/diffusion/bayes_diff"
    exp_dir = f'{EXP_ROOT}/exp_repo_clean/{config.data.dataset}/ddim_fixed_class{args.fixed_class}_train%{args.train_la_data_size}_step{args.timesteps}_S{args.mc_size}_epi_unc_{args.seed}/'
    os.makedirs(exp_dir, exist_ok=True)

    all_sample_x = []
    S, D = last_layers.shape
    for s in range(S):
        sample_x = []

        os.makedirs(exp_dir + f"{s}/", exist_ok=True)

        with torch.no_grad():

            # overwrite the parameters of the last layer with the sampled layer
            model_params = parameters_to_vector(model.parameters())
            model_params[-D:] = last_layers[s]
            vector_to_parameters(model_params, model.parameters())

            for loop in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):

                if config.sampling.cond_class:
                    classes = fixed_classes[:, loop].to(device)
                else:
                    classes = None

                if classes is None:
                    model_kwargs = {}
                else:
                    model_kwargs = {"y": classes}
                
                xT = fixed_xT[loop*args.sample_batch_size:(loop+1)*args.sample_batch_size, :, :, :].to(device)    
                xt_next = xT
                eps_mu_t = model.forward_no_cfg(xT, (torch.ones(args.sample_batch_size) * seq[args.timesteps-1]).to(xT.device).to(torch.int64), **model_kwargs)
        
                for timestep in range(args.timesteps-1, 0, -1):
                    xt_next = singlestep_ddim_sample(betas, xt_next, seq, timestep, eps_mu_t)                
                    eps_mu_t = model.forward_no_cfg(xt_next, (torch.ones(args.sample_batch_size) * seq[timestep-1]).to(xt_next.device).to(torch.int64), **model_kwargs)
    
                x = inverse_data_transform(config, xt_next) 

                torch.save(x.cpu().numpy(), os.path.join(exp_dir + f"{s}/", f"{loop}.pt")) 


if __name__ == "__main__":
    args, config = parse_args_and_config()
    main(args, config)