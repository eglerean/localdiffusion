# Local Diffusion Models

A hands-on project for exploring text-to-image diffusion models locally using [Hugging Face Diffusers](https://huggingface.co/docs/diffusers).

## Setup

```bash
pip install -r requirements.txt
```

Then launch Jupyter from the **project root** (important for model path resolution):

```bash
jupyter lab
```

## Models

| Notebook | Model | Size | CPU inference |
|---|---|---|---|
| `01_tiny_sd.ipynb` | `segmind/tiny-sd` | ~500 MB | ~30–60 s |
| `02_small_sd.ipynb` | `segmind/small-sd` | ~800 MB | ~2–5 min |
| `03_sd_v1_5.ipynb` | `runwayml/stable-diffusion-v1-5` | ~4 GB | ~10–30 min |

Models are downloaded on first use and cached in the `models/` subfolder.  
To free disk space: `rm -rf models/`

## GPU acceleration

If you have a CUDA-capable GPU, uncomment the `pipe.to("cuda")` line in any notebook. This reduces inference time to seconds.

## Quick start

```python
import os
from diffusers import StableDiffusionPipeline
import torch

MODELS_DIR = os.path.abspath("models")
pipe = StableDiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",
    torch_dtype=torch.float32,
    cache_dir=MODELS_DIR,
)
image = pipe("a cat wearing a top hat", num_inference_steps=15, guidance_scale=5.0).images[0]
image.save("output.png")
```
