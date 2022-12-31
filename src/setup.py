from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("VERSION", "r", encoding="utf-8") as f:
    version = f.read()

setup(
    name='src',
    version=version,
    description='CREED implementation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Artem Kudisov',
    author_email='akudisov@ispras.ru',
    maintainer='Artem Kudisov',
    maintainer_email='akudisov@ispras.ru',
    packages=find_packages(include=['src']),
    install_requires=[
        'numpy==1.23.5',
        'torch==1.13.0',
        'transformers==4.23.1',
        'tqdm==4.64.1',
        'scikit-learn==1.2.0'
    ],
    entry_points={
        'creed.plugins': ['src = src']
    },
    data_files=[('', ['VERSION'])],
    python_requires='>=3.10',
    license='MIT License',
)
