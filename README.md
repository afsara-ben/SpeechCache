# Leveraging cache to enable SLU on tiny devices

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
