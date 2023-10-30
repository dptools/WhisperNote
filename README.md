# WhisperNote

A simple Python script to Transcribe audio and perform Speaker Diarization using OpenAI's Whisper and pyannote.audio.

Based on [Majdoddin](https://github.com/Majdoddin)'s work discussed on [GitHub](https://github.com/openai/whisper/discussions/264) and available as a [Google Colab Notebook](https://colab.research.google.com/drive/1HuvcY4tkTHPDzcwyVH77LCh_m8tP-Qet?usp=sharing).

## Running the script

This Project was tested only on Linux, using CPU only and GPU configurations. While it is expected to work on other platforms, it is not guaranteed.

## Benchmarks

Input File: 10 minutes of audio in .mp3, of an interview between 2 people.

### Transcription

CPU: 2.36 minutes
GPU: 2.05 minutes

### Citations

[pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
[pyannote/segmentation](https://huggingface.co/pyannote/segmentation)

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Address = {Brno, Czech Republic},
  Month = {August},
  Year = {2021},
}
``````
