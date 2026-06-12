from diffusers import StableDiffusionPipeline
import torch

access_token="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"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=access_token,
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")


prompt = input("Enter text to create image : ")


image = pipe(prompt).images[0]


image.save("generated_image.png")

print("Image generated and saved as 'generated_image.png'")
