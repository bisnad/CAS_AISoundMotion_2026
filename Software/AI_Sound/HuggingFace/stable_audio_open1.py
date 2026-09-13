import torch
import soundfile as sf
from diffusers import StableAudioPipeline

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device')

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "A Voice that sounds like Creaking Wood"
negative_prompt = "Low quality."

generator = torch.Generator(device).manual_seed(0)

audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=1,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
sf.write("wood.wav", output, pipe.vae.sampling_rate)