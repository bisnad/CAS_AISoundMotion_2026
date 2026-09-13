"""
AudioLDM2: Text-to-Audio Generation with Latent Diffusion Model

Experimenting with Pipeline

code based on: 
    https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2
"""

import scipy
import torch
from diffusers import AudioLDM2Pipeline
import simpleaudio as sa

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

repo_id = "cvssp/audioldm2"
#repo_id = "cvssp/audioldm2-large"
#repo_id = "cvssp/audioldm2-music"
#pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

"""
Using AudioLDM2 for Sound Generation
"""

# define the prompts
prompt = "A Voice that sounds like Creaking Wood"
negative_prompt = "Low quality."
num_inference_steps = 20
audio_length_in_s = 5.0

# set the seed for generator
generator = torch.Generator(device).manual_seed(0)

# run the generation
audios = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    audio_length_in_s=audio_length_in_s,
    num_waveforms_per_prompt=1,
    generator=generator,
).audios


# play audio
sa.play_buffer(audios[0], 1, 4, 16000)

# save the best audio sample (index 0) as a .wav file
scipy.io.wavfile.write("wood.wav", rate=16000, data=audios[0])

"""
Encodes the prompt into text encoder hidden states.
"""

# define the prompts
prompt = "Violin"
negative_prompt = "Low quality."

# Get text embedding vectors
prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
    prompt=prompt,
    negative_prompt=negative_prompt,
    device=device,
    do_classifier_free_guidance=False,
    num_waveforms_per_prompt=1
)

# Pass text embeddings to pipeline for text-conditional audio generation
audios = pipe(
    prompt_embeds=prompt_embeds,
    attention_mask=attention_mask,
    generated_prompt_embeds=generated_prompt_embeds,
    num_inference_steps=num_inference_steps,
    audio_length_in_s=audio_length_in_s,
).audios

# play audio
sa.play_buffer(audios[0], 1, 4, 16000)

# save generated audio sample
scipy.io.wavfile.write("violin.wav", rate=16000, data=audios[0])


"""
Mix Prompt Embeddings
"""

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
    audio_length_in_s=audio_length_in_s,
).audios

# play audio
sa.play_buffer(audios_mix[0], 1, 4, 16000)

# save generated audio sample
scipy.io.wavfile.write("wood_violin_mix.wav", rate=16000, data=audios_mix[0])
