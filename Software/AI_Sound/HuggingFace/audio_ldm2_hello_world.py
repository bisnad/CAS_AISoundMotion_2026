# -------------------------------------------------------------------------------------------------
# AudioLDM2: Text-to-Audio Generation with Latent Diffusion Model
# 
# Experimenting with Pipeline
# 
# code based on: 
#    https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline

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

repo_id = "cvssp/audioldm2"
#repo_id = "cvssp/audioldm2-large"
#repo_id = "cvssp/audioldm2-music"
#pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
#pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32, revision="refs/pr/5")
pipe = pipe.to(device)

# Apple MPS cannot handle the large Conv1d output used by the HiFi-GAN vocoder.
# Keep diffusion on MPS, but run waveform decoding on the CPU.
if device == "mps":
    pipe.vocoder.to("cpu")

    def mel_spectrogram_to_waveform(mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = pipe.vocoder(mel_spectrogram.to("cpu"))
        return waveform.cpu().float()

    pipe.mel_spectrogram_to_waveform = mel_spectrogram_to_waveform

# -------------------------------------------------------------------------------------------------
# Seed Settings
# -------------------------------------------------------------------------------------------------

# set the seed for generator
manual_seed = 42

# -------------------------------------------------------------------------------------------------
# Create Audio from Text Prompts
# -------------------------------------------------------------------------------------------------

# define the prompts
prompts = ["Violin", "Creaking Wood", "A Violin that sounds like Creaking Wood"]
negative_prompt = "Low quality."
num_inference_steps = 20
guidance_scale = 3.5
audio_length_in_s = 5.0

for prompt in prompts:

    if manual_seed is not None:
        generator = torch.Generator(device).manual_seed(manual_seed)

    # run the generation
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        audio_length_in_s=audio_length_in_s,
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios


    # save the first audio sample (index 0) as a .wav file
    output = audio[0]
    sf.write("{}text2audio_{}.wav".format(save_audio_path, prompt), output, 16000)

# -------------------------------------------------------------------------------------------------
# Create Audio from Text Prompt Encodings
# -------------------------------------------------------------------------------------------------

# define the prompts
prompts = ["Violin", "Creaking Wood", "A Violin that sounds like Creaking Wood"]
negative_prompt = "Low quality."

for prompt in prompts:
    if manual_seed is not None:
        generator = torch.Generator(device).manual_seed(manual_seed)

    (
        prompt_embeds,
        attention_mask,
        negative_prompt_embeds,
        negative_attention_mask,
        generated_prompt_embeds,
        negative_generated_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        do_classifier_free_guidance=True,
        num_waveforms_per_prompt=1,
    )

    audio = pipe(
        prompt_embeds=prompt_embeds,
        attention_mask=attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_attention_mask=negative_attention_mask,
        generated_prompt_embeds=generated_prompt_embeds,
        negative_generated_prompt_embeds=negative_generated_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        audio_length_in_s=audio_length_in_s,
        generator=generator,
    ).audios

    # save generated audio sample
    output = audio[0]
    sf.write("{}textembed2audio_{}.wav".format(save_audio_path, prompt), output, 16000)

# -------------------------------------------------------------------------------------------------
# Create Audio from Mixed Text Prompt Encodings
# -------------------------------------------------------------------------------------------------

# define the prompts
prompt1 = "Creaking Wood"
prompt2 = "Voice"
negative_prompt = "Low quality."

# Get text embedding vectors for prompt1
prompt_embed1, attention_mask1, generated_prompt_embeds1 = pipe.encode_prompt(
    prompt=prompt1,
    negative_prompt=negative_prompt,
    device=device,
    do_classifier_free_guidance=False,
    num_waveforms_per_prompt=1
)

# Get text embedding vectors for prompt2
prompt_embed2, attention_mask2, generated_prompt_embeds2 = pipe.encode_prompt(
    prompt=prompt2,
    negative_prompt=negative_prompt,
    device=device,
    do_classifier_free_guidance=False,
    num_waveforms_per_prompt=1
)

# mix the two text embeddings
prompt_embeds_mix = torch.cat((prompt_embed1, prompt_embed2), dim=1)
attention_masks_mix = torch.cat((attention_mask1, attention_mask2), dim=1)
generated_prompt_embeds_mix = 0.5 * generated_prompt_embeds1 + 0.5 * generated_prompt_embeds2

# generate audio with mixed text embedding
audios_mix = pipe(
    prompt_embeds=prompt_embeds_mix,
    attention_mask=attention_masks_mix,
    generated_prompt_embeds=generated_prompt_embeds_mix,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    audio_length_in_s=audio_length_in_s,
).audios

# save generated audio sample
output = audio[0]
sf.write("{}textembedmix2audio_{}_{}.wav".format(save_audio_path, prompt1, prompt2), output, 16000)
