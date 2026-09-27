from pathlib import Path

from setuptools import find_packages
from setuptools import setup

ROOT = Path(__file__).resolve().parent
readme_path = ROOT / "README.md"
if not readme_path.exists():
    readme_path = ROOT / "_README.txt"

with readme_path.open(encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="unloop",
    version="0.0.1",
    python_requires=">=3.13",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.13",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="Generative Music Modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hugo Flores García, Prem Seetharaman",
    author_email="hugggofloresgarcia@gmail.com",
    url="https://github.com/hugofloresgarcia/unloop",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "argbind>=0.3.2",
        "numpy>=2.1.0",
        "wavebeat @ git+https://github.com/hugofloresgarcia/wavebeat",
        "gradio", 
        "loralib",
        "torch_pitch_shift",
        "plotly",
        "pydantic==2.10.6",
        "spaces",
    ],
)
