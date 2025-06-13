from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    """Read README for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="pasd",
    version="0.1.0",
    author="Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, Lei Zhang",
    author_email="yangtao9009@gmail.com",
    description="Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yangxy/PASD",
    project_urls={
        "Paper": "https://arxiv.org/abs/2308.14469",
        "Bug Reports": "https://github.com/yangxy/PASD/issues",
        "Source": "https://github.com/yangxy/PASD",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "test": ["salesforce-lavis==1.0.2"],
        "dev": ["black", "flake8", "isort", "pytest"],
        "all": ["salesforce-lavis==1.0.2", "black", "flake8", "isort", "pytest"],
    },
    include_package_data=True,
    package_data={
        "pasd": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "stable-diffusion",
        "super-resolution", 
        "image-enhancement",
        "deep-learning",
        "computer-vision",
        "pytorch",
        "diffusion-models",
    ],
)