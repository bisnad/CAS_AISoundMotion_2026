import torch
import soundfile as sf
from diffusers import StableAudioPipeline, DPMSolverMultistepScheduler

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using {device} device')

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

original_decode = pipe.vae.decode
def patched_decode(z, *args, **kwargs):
    return original_decode(z.to("cpu"), *args, **kwargs)
pipe.vae.decode = patched_decode
pipe.vae = pipe.vae.to("cpu")

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