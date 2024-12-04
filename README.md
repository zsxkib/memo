# MEMO

**MEMO: Memory-Guided Diffusion for Expressive Talking Video Generation**
<br>
[Longtao Zheng](https://ltzheng.github.io)\*,
[Yifan Zhang](https://scholar.google.com/citations?user=zuYIUJEAAAAJ)\*,
[Hanzhong Guo](https://scholar.google.com/citations?user=q3x6KsgAAAAJ)\,
[Jiachun Pan](https://scholar.google.com/citations?user=nrOvfb4AAAAJ),
[Zhenxiong Tan](https://scholar.google.com/citations?user=HP9Be6UAAAAJ),
[Jiahao Lu](https://scholar.google.com/citations?user=h7rbA-sAAAAJ),
[Jiahao Lu](https://scholar.google.com/citations?user=h7rbA-sAAAAJ),
[Chuanxin Tang](https://scholar.google.com/citations?user=3ZC8B7MAAAAJ),
[Bo An](https://personal.ntu.edu.sg/boan/index.html),
[Shuicheng Yan](https://scholar.google.com/citations?user=DNuiPHwAAAAJ)
<br>
_[Project Page](https://memoavatar.github.io) | [arXiv]() |
[Model](https://huggingface.co/memoavatar/memo) |
[Data](https://huggingface.co/memoavatar/memo-data)_

This repository contains the example inference script for the MEMO-preview model. The gif demo below is compressed. See our [project page](https://memoavatar.github.io) for full videos.

![](assets/demo.gif)


## Installation

```bash
conda create -n memo python=3.10 -y
conda activate memo
conda install -c conda-forge ffmpeg -y
pip install -e .
```

> Our code will download the checkpoint from Hugging Face automatically, and the models for face analysis and vocal separation will be downloaded to `misc_model_dir` of `configs/inference.yaml`. If you want to download the models manually, please download the checkpoint from [here](https://huggingface.co/memoavatar/memo) and specify the path in `model_name_or_path` of `configs/inference.yaml`.

## Inference

```bash
python inference.py --config configs/inference.yaml --input_image <IMAGE_PATH> --input_audio <AUDIO_PATH> --output_dir <SAVE_PATH>
```

For example:

```bash
python inference.py --config configs/inference.yaml --input_image assets/examples/dicaprio.jpg --input_audio assets/examples/speech.wav --output_dir outputs
```

> We tested the code on both H100 and RTX 4090 GPUs with CUDA 12. The inference time is around 1s per frame on H100 and 2s per frame on RTX 4090.

## Citation

If you find our work useful, please use the following citation:

```bibtex
@article{zheng2024memo,
  title={MEMO: Memory-Guided Diffusion for Expressive Talking Video Generation},
  author={Longtao Zheng and Yifan Zhang and Hanzhong Guo and Jiachun Pan and Zhenxiong Tan and Jiahao Lu and Chuanxin Tang and Bo An and Shuicheng Yan},
  journal={arXiv preprint arXiv:2411.xxxxx},
  year={2024}
}
```
