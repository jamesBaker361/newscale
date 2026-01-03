from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import parse_args,default_parser,repo_api_init
from experiment_helpers.image_helpers import concat_images_horizontally
from experiment_helpers.data_helpers import split_data
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.loop_decorator import optimization_loop
from diffusers import UNet2DConditionModel,AutoencoderKL,DiffusionPipeline,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from transformers import CLIPTokenizer,CLIPTextModel
import time
from data_helpers import AFHQDataset
import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn.functional as F
import os
import random
import torchvision.transforms as T
from PIL import Image
import wandb

DISCRETE_SCALE="discrete"
CONTINUOUS_SCALE="continuous"
CONTINUOUS_NOISE="noise"
UNET="unet"
LORA="lora"
CONTROLNET="control"
IPADAPTER="adapter"

def inference(unet:UNet2DConditionModel,
              text_encoder:CLIPTextModel,
              tokenizer:CLIPTokenizer,
              vae:AutoencoderKL,
              image_processor:VaeImageProcessor,
              scheduler:DDIMScheduler,
              #input_latents:torch.Tensor,
              num_inference_steps:int,
              args:dict,
              captions:str,
              device,
              bsz:int,
              dims:list,
              mask:torch.Tensor=None,
              src_image:Image.Image=None #for super resolution
              ):
    if args.text_conditional:
        if type(captions)==str:
            captions=[captions]
            bsz=1
        token_ids= tokenizer(
                    captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
        encoder_hidden_states=text_encoder(token_ids, return_dict=False)[0]
    else:
        encoder_hidden_states=None
        
    latents=None
    if src_image:
        latents=vae.encode(image_processor.preprocess(src_image).to(device)).latent_dist.sample()*vae.config.scaling_factor
        if mask:
            latents=mask*latents
    
    if args.timesteps==CONTINUOUS_NOISE:
        if latents is None:
            latents=torch.randn((bsz,4,args.dim//8,args.dim//8),device=device)

        timesteps,num_inference_steps=retrieve_timesteps(scheduler,num_inference_steps,device=device)
        
    else:
        img = [Image.new("RGB", (args.dim, args.dim), tuple(random.randint(0, 255) for _ in range(3))) for _ in range(bsz)]
        if latents is None:
            latents=vae.encode(image_processor.preprocess(img).to(device)).latent_dist.sample()*vae.config.scaling_factor

        
        if args.timesteps==CONTINUOUS_SCALE:
            timesteps,num_inference_steps=retrieve_timesteps(scheduler,num_inference_steps,device=device)
            
        if args.timesteps==DISCRETE_SCALE:
            timesteps=[torch.tensor(t).long() for t in dims]
            
    for i,t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        #latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        #latent_model_input = scheduler.scale_model_input(latents, t)
        
        noise_pred = unet(
                latents,
                t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, )[0]
    image = image_processor.postprocess(image,)
    return image
        
        
        
    



def main(args):
    api,accelerator,device=repo_api_init(args)
    pipe=DiffusionPipeline.from_pretrained("Lykon/dreamshaper-8")
    vae=pipe.vae.to(device)
    text_encoder=pipe.text_encoder.to(device)
    scheduler=DDIMScheduler()
    tokenizer = CLIPTokenizer.from_pretrained(
        "Lykon/dreamshaper-8", subfolder="tokenizer"
    )
    
    image_processor=VaeImageProcessor()
    train_dataset=AFHQDataset(split="train")
    test_dataset=AFHQDataset(split="val")
    
    # Split the dataset
    train_loader,_,val_loader=split_data(train_dataset,0.9,args.batch_size)
    test_loader=DataLoader(test_dataset)
    for batch in train_loader:
        break
    
    
    if args.method==UNET:
        unet=UNet2DConditionModel().to(device)
        params=[p for p in unet.parameters()]
        
        optimizer=torch.optim.AdamW(params,args.lr)
        model_dict={
        "pytorch_weights.safetensors":unet
        }

        optimizer,train_loader,test_loader,val_loader,vae,unet,text_encoder=accelerator.prepare(
            optimizer,train_loader,test_loader,val_loader,vae,unet,text_encoder)
    
    save_subdir=os.path.join("scale",args.repo_id)
    save,load=save_and_load_functions(model_dict,save_subdir,api,args.repo_id)
    
    start_epoch=load(False)
    
    accelerator.print("starting at ",start_epoch)
    
    #FID Heusel (2017) and IS Salimans (2016). For zero-shot editing tasks we report: LPIPS Zhang et al. (2018) and FID for in/out-painting, and PSNR and SSIM for super-resolution
    
    metrics={
        "psnr":0,
        "ssim":0,
        "lpips":0,
        "fid":0,
        "fid_heusel":0,
        "is_salimans":0
    }
    
    dims=[1]
    while dims[-1]!=args.dim:
        dims.append(2*dims[-1])
        
    dims=dims[::-1]
    
    mask_outpaint=Image.open(os.path.join("data","datasets","gt_keep_masks","ex64")).convert("L")
    mask_outpaint_pt=T.ToTensor()(mask_outpaint.resize((args.dim//8,args.dim//8)))
    
    mask_inpaint_pt=(1.-mask_outpaint_pt)
    
    @optimization_loop(accelerator,train_loader,args.epochs,args.val_interval,args.limit,val_loader,
                       test_loader,save,start_epoch)
    def batch_function(batch,training,misc_dict):
        images=batch["images"]
        captions=batch["captions"]
        
        bsz=len(images)
        if misc_dict["mode"] in ["train","val"]:
            output_latents=vae.encode(image_processor.preprocess(images).to(device)).latent_dist.sample()
            output_latents*=vae.config.scaling_factor
            if args.text_conditional:
                token_ids= tokenizer(
                    captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                encoder_hidden_states=text_encoder(token_ids, return_dict=False)[0]
            else:
                encoder_hidden_states=None
            
            if args.timesteps==CONTINUOUS_NOISE:
                noise=torch.randn_like(output_latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=output_latents.device)
        
                input_latents=scheduler.add_noise(output_latents,noise,timesteps.long())
            else:
                if args.timesteps==CONTINUOUS_SCALE:
                    scales=[int((args.dim)*random.random()) for r in range(bsz)]
                    scaled_images=[img.resize((args.dim-r,args.dim-r)).resize((args.dim,args.dim)) for r,img in zip(scales,images)]
                    timesteps=torch.tensor([int(scheduler.config.num_train_timesteps*s/(args.dim-1)) for s in scales]).long()
                    
                if args.timesteps==DISCRETE_SCALE:
                    scales=random.choices(len(dims),k=bsz)
                    scaled_images=[img.resize((dims[j],dims[j])).resize((args.dim,args.dim)) for j,img in zip(scales,images)]
                    timesteps=torch.tensor([int(scheduler.config.num_train_timesteps*s/(len(dims))) for s in scales]).long()
                
                input_latents=vae.encode(image_processor.preprocess(scaled_images)).latent_dist.sample()
                
            if training:
                with accelerator.accumulate(params):
                    with accelerator.autocast():
                        
                        predicted=unet(input_latents,timesteps,encoder_hidden_states=encoder_hidden_states,return_dict=False)[0]
                        loss=F.mse_loss(predicted.float(),output_latents.float())
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                predicted=unet(input_latents,timesteps,encoder_hidden_states=encoder_hidden_states,return_dict=False)[0]
                loss=F.mse_loss(predicted.float(),output_latents.float())
        if misc_dict["mode"] in ["test","val"]:
            #inpainting
            gen_inpaint=inference(unet,text_encoder,
                                  tokenizer,vae,
                                  image_processor,scheduler,args.num_inference_steps,args,
                                  captions,device,bsz,dims,mask_inpaint_pt,images)
            
            gen_outpaint=inference(unet,text_encoder,
                                  tokenizer,vae,
                                  image_processor,scheduler,args.num_inference_steps,args,
                                  captions,device,bsz,dims,mask_outpaint_pt,images)
            
            super_res=inference(unet,text_encoder,
                                  tokenizer,vae,
                                  image_processor,scheduler,args.num_inference_steps,args,
                                  captions,device,bsz,dims,None,[i.resize((args.dim//2, args.dim//2)).resize((args.dim,args.dim)) for i in  images])
            
            batch_num=misc_dict["b"]
            count=args.batch_size*batch_num
            
            for k,(inp,outp,super,real) in enumerate(zip(gen_inpaint,gen_outpaint,super_res,images)):
                accelerator.log({
                    f"{misc_dict['mode']}_{k+count}":wandb.Image(concat_images_horizontally([inp,outp,super,real]))
                })

if __name__=='__main__':
    print_details()
    start=time.time()
    parser=default_parser({"project":"scale-test"})
    parser.add_argument("--timesteps",type=str,default=DISCRETE_SCALE,help=f"{DISCRETE_SCALE} or {CONTINUOUS_SCALE} or {CONTINUOUS_NOISE}")
    parser.add_argument("--method",type=str,default=UNET,help=f"one of {UNET} {CONTROLNET} {LORA} {IPADAPTER}")
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--min_dim",type=int,default=1)
    parser.add_argument("--text_conditional",action="store_true")
    parser.add_argument("--num_inference_steps",type=int,default=20)
    args=parse_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")