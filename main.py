from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import parse_args,default_parser,repo_api_init
from experiment_helpers.image_helpers import concat_images_horizontally
from experiment_helpers.data_helpers import split_data
from experiment_helpers.saving_helpers import save_and_load_functions,CONFIG_NAME
from experiment_helpers.loop_decorator import optimization_loop
from diffusers import UNet2DConditionModel,AutoencoderKL,DiffusionPipeline,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.models.attention_processor import  IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor,Attention
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
from huggingface_hub import hf_hub_download
import json
from peft import LoraConfig,get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
import requests
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import Dataset
from torchmetrics.image.inception import InceptionScore
import numpy as np
from tqdm.auto import tqdm

DISCRETE_SCALE="discrete"
CONTINUOUS_SCALE="continuous"
CONTINUOUS_NOISE="noise"
UNET="unet"
LORA="lora"
CONTROLNET="control"
IPADAPTER="adapter"
BASE_REPO="SimianLuo/LCM_Dreamshaper_v7"
VELOCITY="v_prediction"
EPSILON="epsilon"
SAMPLE="sample"

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
              output_type:str,
              mask:torch.Tensor=None,
              src_image:torch.Tensor=None #for super resolution
              ):
    if args.text_conditional:
        if type(captions)==str:
            captions=[captions]*bsz
            #bsz=1
    else:
        captions=[" "]*bsz
    token_ids= tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(device)
    encoder_hidden_states=text_encoder(token_ids, return_dict=False)[0]

        
    latents=None
    if src_image:
        latents=vae.encode(src_image.to(device)).latent_dist.sample()*vae.config.scaling_factor
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
            
    with tqdm(total=num_inference_steps) as progress_bar:
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
            
            progress_bar.update()
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, )[0]
    image = image_processor.postprocess(image,output_type=output_type)
    return image
        
        
        
    



def main(args):
    api,accelerator,device=repo_api_init(args)
    if args.cpu:
        device=torch.device("cpu")
    print("device ",device)
    pipe=DiffusionPipeline.from_pretrained(BASE_REPO)
    vae=pipe.vae.to(device)
    text_encoder=pipe.text_encoder.to(device)
    scheduler=DDIMScheduler(prediction_type=args.prediction_type)
    scheduler.set_timesteps(args.num_inference_steps)
    tokenizer = CLIPTokenizer.from_pretrained(
        BASE_REPO, subfolder="tokenizer"
    )
    
    image_processor=VaeImageProcessor()
    train_dataset=AFHQDataset(split="train")
    test_dataset=AFHQDataset(split="val")
    
    # Split the dataset
    train_loader,_,val_loader=split_data(train_dataset,0.9,args.batch_size)
    test_loader=DataLoader(test_dataset)
    for batch in train_loader:
        break
    save_subdir=os.path.join("scale",args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)
    
    
    
    if args.method==LORA:
        unet=pipe.unet
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet=get_peft_model(unet,peft_config=unet_lora_config)
        config_dict={
            "train":{
            "start_epoch":1
            }
        }
        weights_name="lora_weights.safetensors"
        def save():
            state_dict=get_peft_model_state_dict(unet)
            
            save_path=os.path.join(save_subdir,weights_name)
            torch.save(state_dict,save_path)
            try:
                api.upload_file(path_or_fileobj=save_path,
                                        path_in_repo=weights_name,
                                        repo_id=args.repo_id)
                print(f"uploaded {args.repo_id} to hub")
            except Exception as e:
                print(f"failed to upload {weights_name}")
                print(e)
            config_path=os.path.join(save_subdir, CONFIG_NAME)
            with open(config_path,"w+") as config_file:
                config_dict["train"]["start_epoch"]+=1
                json.dump(config_dict,config_file, indent=4)
                pad = " " * 2048  # ~1KB of padding
                config_file.write(pad)
            try:
                api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                        repo_id=args.repo_id)
            except Exception as e:
                print(f"failed to upload {CONFIG_NAME}")
                print(e)
                
        def load(hf:bool):
            start_epoch = 1  # fresh training
            try:
                if hf:
                    index_path = hf_hub_download(args.repo_id, CONFIG_NAME)
                    pretrained_weights_path=api.hf_hub_download(args.repo_id,weights_name,force_download=True)
                else:
                    index_path = os.path.join(save_subdir, CONFIG_NAME)
                    pretrained_weights_path=os.path.join(save_subdir,weights_name,)
                    
                with open(index_path, "r") as f:
                    data = json.load(f)
                if "train" in data and "start_epoch" in data["train"]:
                    start_epoch = data["train"]["start_epoch"]
                    
                set_peft_model_state_dict(unet,torch.load(pretrained_weights_path))
                    
                return start_epoch
            except FileNotFoundError as err:
                print("file not found",err)
            except requests.exceptions.HTTPError as err:
                print("http error",err)
                
            return start_epoch
        
        params=[p for p in unet.parameters() if p.requires_grad]
        
        
    else:
        if args.method==UNET:
            config_path=hf_hub_download(BASE_REPO,"config.json",subfolder="unet")
            
            unet=UNet2DConditionModel.from_config(json.load(open(config_path,"r"))).to(device)
            #unet=pipe.unet
            params=[p for p in unet.parameters()]
            
            
            model_dict={
            "pytorch_weights.safetensors":unet
            }

            '''optimizer,train_loader,test_loader,val_loader,vae,unet,text_encoder=accelerator.prepare(
                optimizer,train_loader,test_loader,val_loader,vae,unet,text_encoder)'''
        elif args.method==IPADAPTER:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            unet=pipe.unet
            unet.requires_grad_(False)
            unet.image_encoder.requires_grad_(True)
            for attn_name, attn_processor in unet.attn_processors.items():
                if isinstance(
                    attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor,)
                ):
                    attn_processor.requires_grad_(True)
            params=[p for p in unet.parameters()]
            
            
        save,load=save_and_load_functions(model_dict,save_subdir,api,args.repo_id)
        
    print("len params",len(params))
    optimizer=torch.optim.AdamW(params,args.lr)
            
            
        
    
    #save,load=save_and_load_functions(model_dict,save_subdir,api,args.repo_id)
    
    start_epoch=load(False)
    
    accelerator.print("starting at ",start_epoch)
    
    #FID Heusel (2017) and IS Salimans (2016). For zero-shot editing tasks we report: LPIPS Zhang et al. (2018) and FID for in/out-painting, and PSNR and SSIM for super-resolution
    
    test_metric_dict={
        "psnr":0,
        "ssim":0,
        "lpips_out":0,
        "fid_out":0,
        "lpips_in":0,
        "fid_in":0,
        "fid_gen":0,
        "incept":0
    }
    
    ssim_metric=StructuralSimilarityIndexMeasure(data_range=(-1.0,1.0))
    psnr_metric=PeakSignalNoiseRatio(data_range=(-1.0,1.0))
    lpips_metric=LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    fid_metric=FrechetInceptionDistance(feature=64,normalize=False) #expects images in [0,255]
    inception_metric=InceptionScore(normalize=False) #expects images in [0,255]
    
    dims=[1]
    while dims[-1]!=args.dim:
        dims.append(2*dims[-1])
        
    dims=dims[::-1]
    iteration_dims=[1]+[int(args.dim*float(j/args.num_inference_iterations)) for j in range(1,1+args.num_inference_iterations)]
    print("dims",dims)
    print("iteration dims",iteration_dims)
    
    
    mask_outpaint=Image.open(os.path.join("data","datasets","gt_keep_masks","ex64","000000.png")).convert("L")
    mask_outpaint_pt=T.ToTensor()(mask_outpaint.resize((args.dim//8,args.dim//8)))
    
    mask_inpaint_pt=(1.-mask_outpaint_pt)
    
    def normalize(images:torch.Tensor)->torch.Tensor:
        #[-1,1] to [0,255]
        _images=images*128
        _images=_images+128
        _images=_images.to(torch.uint8)
        
        return _images
    
    @optimization_loop(accelerator,train_loader,args.epochs,args.val_interval,args.limit,val_loader,
                       test_loader,save,start_epoch)
    def batch_function(batch,training,misc_dict):
        loss=0.0
        images=batch["image"]
        captions=batch["caption"]
        
        bsz=len(images)
        
        if misc_dict["epochs"]==start_epoch and misc_dict["b"]==0:
            print("imgease size ",images.size())
        if misc_dict["mode"] in ["train","val"]:
            real_latents=vae.encode(images.to(device)).latent_dist.sample()
            real_latents*=vae.config.scaling_factor
            if args.text_conditional:
                token_ids= tokenizer(
                    captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
            else:
                token_ids= tokenizer(
                    [" "]*bsz, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
            encoder_hidden_states=text_encoder(token_ids.to(device), return_dict=False)[0]

            
            if args.timesteps==CONTINUOUS_NOISE:
                noise=torch.randn_like(real_latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=real_latents.device)
        
                input_latents=scheduler.add_noise(real_latents,noise,timesteps.long())
                
            else:
                if args.timesteps==CONTINUOUS_SCALE:
                    scales=[int((args.dim)*random.random()) for r in range(bsz)]
                    #scaled_images=[img.resize((args.dim-r,args.dim-r)).resize((args.dim,args.dim)) for r,img in zip(scales,images)]
                    scaled_images=[T.Resize(args.dim)(T.Resize(args.dim-r)(img)) for r,img in zip(scales,images) ]
                    timesteps=torch.tensor([int(scheduler.config.num_train_timesteps*s/(args.dim-1)) for s in scales]).long()
                    
                if args.timesteps==DISCRETE_SCALE:
                    scales=random.choices([j for j in range(len(dims))],k=bsz)
                    #scaled_images=[img.resize((dims[j],dims[j])).resize((args.dim,args.dim)) for j,img in zip(scales,images)]
                    scaled_images=[T.Resize(args.dim)(T.Resize(dims[j])(img)) for j,img in zip(scales,images) ]
                    timesteps=torch.tensor([int(scheduler.config.num_train_timesteps*s/(len(dims))) for s in scales]).long()
                
                
                input_latents=vae.encode(torch.stack(scaled_images).to(device)).latent_dist.sample()
                print('bsz',bsz,'input_latents.size()',input_latents.size(),'torch.stack(scaled_images)',torch.stack(scaled_images).size(),'real latensts' ,real_latents.size())
                noise=input_latents -real_latents #the "noise"
            if args.prediction_type==EPSILON:
                target_latents=noise
            elif args.prediction_type==SAMPLE:
                target_latents=real_latents
            elif args.prediction_type==VELOCITY:
                target_latents = scheduler.get_velocity(real_latents, noise, timesteps)
                
            if training:
                with accelerator.accumulate(params):
                    with accelerator.autocast():
                        
                        predicted=unet(input_latents,timesteps,encoder_hidden_states=encoder_hidden_states,return_dict=False)[0]
                        loss=F.mse_loss(predicted.float(),target_latents.float())
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                predicted=unet(input_latents,timesteps,encoder_hidden_states=encoder_hidden_states,return_dict=False)[0]
                loss=F.mse_loss(predicted.float(),real_latents.float())
        if misc_dict["mode"] in ["test","val"]:
            #inpainting
            gen_inpaint=inference(unet,text_encoder,
                                  tokenizer,vae,
                                  image_processor,scheduler,args.num_inference_steps,args,
                                  captions,device,bsz,dims, "pt",mask_inpaint_pt,images,)
            
            gen_outpaint=inference(unet,text_encoder,
                                  tokenizer,vae,
                                  image_processor,scheduler,args.num_inference_steps,args,
                                  captions,device,bsz,dims, "pt",mask_outpaint_pt,images)
            
            super_res=inference(unet,text_encoder,
                                  tokenizer,vae,
                                  image_processor,scheduler,args.num_inference_steps,args,
                                  captions,device,bsz,dims, "pt",None,[i.resize((args.dim//2, args.dim//2)).resize((args.dim,args.dim)) for i in  images])
            
            batch_num=misc_dict["b"]
            count=args.batch_size*batch_num
            
            gen_inpaint_pil=image_processor.postprocess(gen_inpaint)
            gen_outpaint_pil=image_processor.postprocess(gen_outpaint)
            super_res_pil=image_processor.postprocess(super_res)
            images_pil=image_processor.postprocess(images)
            
            
            for k,(inp,outp,super,real) in enumerate(zip(gen_inpaint_pil,gen_outpaint_pil,super_res_pil,images_pil)):
                accelerator.log({
                    f"{misc_dict['mode']}_{k+count}":wandb.Image(concat_images_horizontally([inp,outp,super,real]))
                })
            if misc_dict["mode"]=="test":
                ssim_score=ssim_metric(super_res,images) #ssim(preds, target)
                psnr_score=psnr_metric(super_res,images)
                lpips_score_in=lpips_metric(gen_inpaint,images)
                lpips_score_out=lpips_metric(gen_outpaint,images)
                fid_metric.update(normalize(images),True)
                fid_metric.update(normalize(gen_inpaint),False)
                fid_score_in=fid_metric.compute()
                fid_metric.update(normalize(gen_outpaint),False)
                fid_score_out=fid_metric.compute()
                
                for key,score in zip(["ssim","psnr","lpips_in","lpips_out","fid_in","fid_out"],
                                     [ssim_score,psnr_score,lpips_score_in,lpips_score_out,fid_score_in,fid_score_out]):
                    test_metric_dict[key].append(score.cpu().detach().numpy())
    batch_function()
    accelerator.free_memory()
    
    
    
    with torch.no_grad():
        print("generation task beginning")
        start=time.time()
        k=0
        output_dict={
            "image":[],
            "caption":[]
        }
        real_images=torch.cat([batch["image"] for batch in test_loader ])
        gen_images=[]
        while k<args.n_test:
            captions=random.sample([cap for cap in test_dataset.cat_set],args.batch_size)
            images=inference(unet,text_encoder,tokenizer,vae,
                             image_processor,scheduler,args.num_inference_steps,args,
                             captions,device,args.batch_size,dims,"pt",None,None
                             )
            gen_images.append(images)
            images_pil=image_processor.postprocess(images)
            for cap,img in zip(captions,images_pil):
                output_dict["image"].append(img)
                output_dict["caption"].append(cap)
            accelerator.free_memory()
        gen_images=torch.cat(gen_images)
        Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)
        fid_metric.update(normalize(real_images),True)
        fid_metric.update(normalize(gen_images),False)
        test_metric_dict["fid_gen"].append(fid_metric.compute().cpu().detach().numpy())
        
        inception_metric.update(gen_images)
        test_metric_dict["incept"].append(inception_metric.compute())
        
        test_metric_dict={
            key:np.mean(value) for key,value in test_metric_dict.items()
        }
        print(test_metric_dict)
        accelerator.log(test_metric_dict)
        generation_end=time.time()
        
        print("generation task elapsed, ",generation_end-start)
            
    

if __name__=='__main__':
    print_details()
    start=time.time()
    parser=default_parser({"project":"scale-test"})
    parser.add_argument("--timesteps",type=str,default=DISCRETE_SCALE,help=f"{DISCRETE_SCALE} or {CONTINUOUS_SCALE} or {CONTINUOUS_NOISE}")
    parser.add_argument("--method",type=str,default=UNET,help=f"one of {UNET} {CONTROLNET} {LORA} {IPADAPTER}")
    parser.add_argument("--num_inference_iterations",type=int,default=4)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--min_dim",type=int,default=1)
    parser.add_argument("--text_conditional",action="store_true")
    parser.add_argument("--num_inference_steps",type=int,default=20)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--rank",type=int,default=4)
    parser.add_argument("--n_test",type=int,default=5)
    parser.add_argument("--dest_dataset",type=str,default="jlbaker361/test-scale-images")
    parser.add_argument("--prediction_type",type=str,help=f" one of {VELOCITY}, {EPSILON} or {SAMPLE}")
    args=parse_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")