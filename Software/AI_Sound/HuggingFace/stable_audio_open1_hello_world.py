# -------------------------------------------------------------------------------------------------
# Stable Audio Open 1.0: Text-to-Audio Generation with Latent Diffusion Model
# 
# Experimenting with Pipeline
# 
# code based on:
#    https://huggingface.co/docs/diffusers/main/api/pipelines/stable_audio
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
import torch
import soundfile as sf
from diffusers import StableAudioPipeline, DPMSolverMultistepScheduler

# -------------------------------------------------------------------------------------------------
# Compute Unit
# -------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device')

# -------------------------------------------------------------------------------------------------
# Save Paths Settings
# -------------------------------------------------------------------------------------------------

save_audio_path = os.path.join("results/audio/")

os.makedirs(save_audio_path, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Create Pipeline
# -------------------------------------------------------------------------------------------------

repo_id = "stabilityai/stable-audio-open-1.0"
pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Apple MPS cannot handle the Oobleck VAE decode reliably.
# Keep diffusion (transformer) on MPS, but run VAE decoding on the CPU.
if device == "mps":
    original_decode = pipe.vae.decode

    def patched_decode(z, *args, **kwargs):
        return original_decode(z.to("cpu"), *args, **kwargs)

    pipe.vae.decode = patched_decode
    pipe.vae = pipe.vae.to("cpu")

# -------------------------------------------------------------------------------------------------
# Seed Settings
# -------------------------------------------------------------------------------------------------

# set the seed for generator
manual_seed = 42

# -------------------------------------------------------------------------------------------------
# Create Audio from Text Prompts
# -------------------------------------------------------------------------------------------------

prompts = ["Violin", "Creaking Wood", "A Violin that sounds like Creaking Wood"]
negative_prompt = "Low quality."
num_inference_steps = 100
audio_end_in_s = 5.0
guidance_scale = 4.5

for prompt in prompts:

    if manual_seed is not None:
        generator = torch.Generator(device).manual_seed(manual_seed)
    
    # run the generation
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        audio_end_in_s=audio_end_in_s,
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios

    # save the first audio sample (index 0) as a .wav file
    output = audio[0].T.float().cpu().numpy()
    sf.write("{}text2audio_{}.wav".format(save_audio_path, prompt), output, pipe.vae.sampling_rate)

# -------------------------------------------------------------------------------------------------
# Create Audio from Text Prompt Encodings
# -------------------------------------------------------------------------------------------------

# define the prompt
prompts = ["Violin", "Creaking Wood", "A Violin that sounds like Creaking Wood"]
negative_prompt = "Low quality."

for prompt in prompts:

    if manual_seed is not None:
            generator = torch.Generator(device).manual_seed(manual_seed)

    # Get text embedding vector (already projected)
    prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        do_classifier_free_guidance=False,
        negative_prompt=negative_prompt,
    )

    # Pass text embeddings to pipeline for text-conditional audio generation
    audio = pipe(
        prompt_embeds=prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        audio_end_in_s=audio_end_in_s,
        generator=generator,
    ).audios

    # save generated audio sample
    output = audio[0].T.float().cpu().numpy()
    sf.write("{}textembed2audio_{}.wav".format(save_audio_path, prompt), output, pipe.vae.sampling_rate)
    

# -------------------------------------------------------------------------------------------------
# Create Audio from Mixed Text Prompt Encodings
# -------------------------------------------------------------------------------------------------

# define the prompts
prompt1 = "Violin"
prompt2 = "Creaking Wood"
negative_prompt = "Low quality."

if manual_seed is not None:
    generator = torch.Generator(device).manual_seed(manual_seed)

# Get text embedding vector for prompt1
prompt_embeds1 = pipe.encode_prompt(
    prompt=prompt1,
    device=device,
    do_classifier_free_guidance=False,
    negative_prompt=negative_prompt,
)

# Get text embedding vector for prompt2
prompt_embeds2 = pipe.encode_prompt(
    prompt=prompt2,
    device=device,
    do_classifier_free_guidance=False,
    negative_prompt=negative_prompt,
)

# mix the two text embeddings
prompt_embeds_mix = 0.5 * prompt_embeds1 + 0.5 * prompt_embeds2

# generate audio with mixed text embedding
audio_mix = pipe(
    prompt_embeds=prompt_embeds_mix,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    audio_end_in_s=audio_end_in_s,
    generator=generator,
).audios

# save generated audio sample
output = audio_mix[0].T.float().cpu().numpy()
sf.write("{}textembedmix2audio_{}_{}.wav".format(save_audio_path, prompt1, prompt2), output, pipe.vae.sampling_rate)
 