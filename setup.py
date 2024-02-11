from setuptools import setup, find_packages


setup(
    name='DreamBrookPy', 
    version='0.0.1',
    author='DreamBrookTech',
    author_email='dev@dreambrook.tech',
    description='Official Python Toolbox for DreamBrookTech Ecosystem',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DreamBrookTech/DreamBrookPy',
    packages=find_packages(),
    python_requires='>=3.9',
)
