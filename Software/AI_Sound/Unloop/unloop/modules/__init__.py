import audiotools

audiotools.ml.BaseModel.INTERN += ["unloop.modules.**"]
audiotools.ml.BaseModel.EXTERN += ["einops", "flash_attn.flash_attention", "loralib"]

from .transformer import VampNet