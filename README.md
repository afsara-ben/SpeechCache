



# SpeechCache: Speech Understanding on Tiny Devices with A Learning Cache

This repository contains the source code for Mobisys'24 paper *Leveraging cache to enable SLU on tiny devices by authors* by Afsara Benazir, Zhiming Xu, and Felix Xiaozhu Lin.

## Dependencies
The code is compatible with `python 3.10`. A list of dependencies can be installed via `pip install -r requirements.txt`.

## Training
The `.csv` files are located in `SLURP/csv`. It contains a detailed list of entries combining columns from the original SLURP dataset and metadata. 

To run a sample train script, please run `slurp_train.py` as follows:

```
python3 slurp_train.py --model_dir <your_model_path> --wav_path <your_wav_path>
```
## Inference/Testing
The trained models have a metadata file ending in `.pkl.gz` that contains the index of the train samples. During testing, those samples are omitted. 

To do inference, please run
```
python3 slurp_test.py --model_dir <your_model_path> --wav_path <your_wav_path>
```
Optional: To play with the optimizations mentioned in paper, please set `--dynamic True` for doing inference with dynamic threshold (c.f pg7 of manuscript).
set `-in_domain True`` for using a pretrained slurp model (c.f pg7 of manuscript)

## Models

All finetuned SLURP-C models used in the experiment can be found at [here](https://zenodo.org/records/11106484?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI0YjExMDc0LWVmMDUtNDFjOC1hNGQxLTcwZmMzZTg2ZTczMiIsImRhdGEiOnt9LCJyYW5kb20iOiJhMWVkY2VmZTJmYjI0NjRkOTYxNDE1ZmEyZWM1ZDY4MyJ9.ifDZQ3TbMcesQ0x4EJIHtqc4yjpo0OrsGfsl7CdxSc1PUzE_lBIHz2zkHPom1VvX5JaX6NZTAzSYBgacwJYCrA)

Models for user study (in the wild evaluation) are [here](https://zenodo.org/records/11106505?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA3Y2NkNjhmLTAwOGItNDU3Zi05MDg3LWVkMDI3YjE4MTAxOCIsImRhdGEiOnt9LCJyYW5kb20iOiIxYzRhYTkzZTdjZTk5ZjQ3MWZiY2E5M2Y2NTczYmQ4YiJ9.AedUn184TwA0zyAJXSTajOsAiTmHsu3CbyD0imJGeWOiH7UO0aVjb-0RAkZX9_nAhVteASBdHFapd8mJr_TgqA)

## MCU Code
Step 1: Pytorch models cannot directly run on the target hardware, we create two dummy models that represent the CNN and GRU blocks of SC using `STM32CubeIDE/pytorch_to_keras.ipynb` and convert it to keras format, saved under `STM32CubeIDE/mcu_models` (files conv1d_3_model.h5 and gru_2_model.h5).

Step 2: Open `STM32CubeIDE/workspace_1.13.1/test-rnn-no-peripherals-rnn` in STMCubeIDE after performing necessary installations. `test-rnn-no-peripherals-rnn.ioc` is the main file that will run on the target hardware/simulator. Under Middleware and Software tab, open X-CUBE-AI, configure necessary items and click Analyze. The output of debug console and X-CUBE-AI is in `latency numbers.pdf`.

## Demo

https://github.com/afsara-ben/SpeechCache/assets/44926095/b0bb85ff-046f-49e5-b4c5-e14da186aad7
## Reference
```bibtex
@article{benazir2023leveraging,
  title={Leveraging cache to enable SLU on tiny devices},
  author={Benazir, Afsara and Xu, Zhiming and Lin, Felix Xiaozhu},
  journal={arXiv preprint arXiv:2311.18188},
  year={2023}
}
```
Please cite our paper if you find our work useful.

## Acknowlegment
The code is adapted based on [end-to-end-SLU](https://github.com/lorenlugosch/end-to-end-SLU/tree/master), published in *Speech Model Pre-training for End-to-End Spoken Language Understanding* by Lugosch et al.
