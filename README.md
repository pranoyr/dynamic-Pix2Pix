
## Dynamic-Pix2Pix: Noise Injected cGAN for Modeling Input and Target Domain Joint Distributions with Limited Training Data

[[Paper](https://arxiv.org/pdf/2211.08570.pdf)] 


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

## References
https://arxiv.org/pdf/2211.08570.pdf

## License
This project is licensed under the MIT License