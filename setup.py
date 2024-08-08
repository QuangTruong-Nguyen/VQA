from setuptools import setup, find_packages

setup(
    name="project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "datasets",
        "opencv-python-headless",
        "accelerate",
        "peft",
        "bitsandbytes",
        "einops",
        "timm",
        "tiktoken",
        "wandb"
    ],
)
