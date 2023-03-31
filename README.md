
## Dynamic-Pix2Pix: Noise Injected cGAN for Modeling Input and Target Domain Joint Distributions with Limited Training Data

[[Paper](https://arxiv.org/pdf/2211.08570.pdf)] 


## Table of Contents
  * [License](#license)
  * [Description](#description)
  * [News](#news)
  * [Installation](#installation)
  * [Downloading the model](#downloading-the-model)
  * [Loading SMPL-X, SMPL+H and SMPL](#loading-smpl-x-smplh-and-smpl) 
    * [SMPL and SMPL+H setup](#smpl-and-smplh-setup)
    * [Model loading](https://github.com/vchoutas/smplx#model-loading)
  * [MANO and FLAME correspondences](#mano-and-flame-correspondences) 
  * [Example](#example)
  * [Modifying the global pose of the model](#modifying-the-global-pose-of-the-model)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)


## Download Dataset 
```
scripts/edges2shoes.sh
```

## Train
```
python main.py --cfg cfg/exp1.yaml
```


## Inference
```
python inference/inference.py --img_path <image-path> --checkpoint <path-to-checkpoint>
```


## License
This project is licensed under the MIT License