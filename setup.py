from setuptools import setup, find_packages

setup(
    name="dpsniper",
    version="0.1.0",
    author="SRI Lab ETH Zurich",
    description="A machine-learning-based tool for discovering differential privacy violations in black-box algorithms",
    packages=find_packages(exclude=["eval_sp2021","statdp","statdpwrapper","tests"]),
    python_requires='>=3.8',
    install_requires=[
        'matplotlib>=3.2.0',
        'pandas>=1.1.0',
        'numpy>=1.19.0',
        'torch>=1.8.0',
        'tensorboard>=2.3.0',
        'scikit-learn>=0.22.1',
        'mmh3>=2.5.1'
    ]
)
