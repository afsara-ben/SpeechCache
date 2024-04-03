# Leveraging cache to enable SLU on tiny devices

This repository contains the source code for the paper titled "Leveraging cache to enable SLU on tiny devices" by authors - Afsara Benazir (co-author), Zhiming Xu (co-author), Felix Xiaozhu Lin.

## Dependencies
The code is compatible with `python 3.10`. A list of dependencies can be found in `requirements.txt`.

## Training
The `.csv` files are located in `SLURP/csv`. It contains a detailed list of entries combining columns from the original SLURP dataset+metadata. 

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


## Acknowlegment
This work is inspired by the following end to end SLU project by Lugosch et. al.
* [end-to-end-SLU](https://github.com/lorenlugosch/end-to-end-SLU/tree/master): "Speech Model Pre-training for End-to-End Spoken Language Understanding"


If you find our paper useful, you can cite us:
```bibtex
@article{benazir2023leveraging,
  title={Leveraging cache to enable SLU on tiny devices},
  author={Benazir, Afsara and Xu, Zhiming and Lin, Felix Xiaozhu},
  journal={arXiv preprint arXiv:2311.18188},
  year={2023}
}
```
