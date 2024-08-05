<p align="center">
    <img src="logo.png" alt="drawing" width="250" />
</p>


# AudioFlex: Python Library for Audio Stretching

## Introduction
Welcome to AudioFlex, a comprehensive Python library dedicated to audio stretching and manipulation. This project stems from a gap I found in the realm of audio processing tools – the lack of a pure Python library for intricate audio stretching algorithms. AudioFlex is developed to demystify the complexities often shrouded by extensive C++ implementations, making these algorithms accessible and understandable to a broader audience.

## Example
```python
from AudioIO.input_streams import SineWaveStream
from AudioIO.output_streams import AudioPlaybackProcessor
from audioflex.wsola import WSOLA


sound = SineWaveStream(150, 1, chunk_size=1024, channels=2)
wsola = WSOLA(channels=sound.channels, chunk_size=sound.chunk_size, block_size=512, search_range=128)
speaker = AudioPlaybackProcessor(sound.chunk_size, sound.sample_rate, channels=2)
for chunk in sound.iterable():
    chunk = wsola.process(chunk, rate=1)
    speaker.process(chunk)

```

## Motivation
The inspiration for AudioFlex originated from the need for a Python-centric approach to audio stretching. Existing solutions heavily rely on C++ code, creating a barrier for those seeking to understand the underlying mechanics. This project aims to simplify this process, offering a suite of algorithms in Python that are both educational and functional.
In the future I want to create some music software / instruments using these algorithms.

## Features
- **Pure Python Implementation:** Unlike other libraries that depend on C++ bindings, AudioFlex is written entirely in Python, ensuring ease of understanding and modification.
- **Comprehensive Algorithm Suite:** The library includes various algorithms for audio stretching, pitch shifting, and time-scaling, providing users with multiple tools for audio manipulation.
- **Educational Resource:** The algorithms are presented in a way that emphasizes learning and understanding, making this library an excellent resource for students and enthusiasts alike.

## Reference Material
This project heavily references the work and teachings of AudioSmith, particularly the lecture "Four Ways To Write A Pitch-Shifter - Geraint Luff - ADC22." The lecture and accompanying C++ code have been instrumental in shaping the understanding and implementation of the algorithms in this library. For a deeper dive into the topic, we highly encourage watching the lecture [here](https://www.youtube.com/watch?v=fJUmmcGKZMI&t=569s).

### Requirements
Please ensure you have the necessary dependencies installed. A `requirements.txt` file is included for convenience. Install the dependencies using:
```
pip install -r requirements.txt
```

Dependencies:
- numpy (version ~1.26.3)
- scipy (version ~1.11.4)

## Usage and Modules
AudioFlex is designed for ease of use with detailed documentation. Key modules include:

- **wsola.py**: Implements the WSOLA algorithm for time-scale modification.
- **buffer_handlers.py**: Manages audio data efficiently using numpy arrays.
- **overlap_add.py**: Incorporates the Overlap-Add method for audio stretching.
- And more in the makes...

## Contribution
Contributions to AudioFlex are welcomed and appreciated. Whether it's improving the code, adding new features, or enhancing documentation, your input is appreciated.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to AudioSmith and Geraint Luff for their invaluable insights and contributions to the field of audio processing. Their work has been a guiding light in the development of this library.
