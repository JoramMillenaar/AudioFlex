from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='AudioFlex',
    description='A Pure Python Suite of Algorithms to Stretch Audio Duration without Affecting Pitch',
    version='0.0.3',
    author='Joram Millenaar',
    author_email='joormillenaar@live.nl',
    url='https://github.com/jofoks/AudioFlex',
    install_requires=['numpy', 'scipy'],
    packages=['audioflex'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'duration', 'pitch', 'music', 'stretch', 'wsola', 'overlap-add'],
    license='MIT'
)
