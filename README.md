# GSNET-pytorch
Pytorch implementation for science paper: <br>
* Generalized Stochastic Neighbor Embedding Training for Pattern Discovery
## Usage
1. install [pytorch >= 1.0.0](https://pytorch.org/get-started/locally/) and the matched cuda version <br>
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch 
```

2. Clone the code to local <br>
```    
git clone https://github.com/zzynb97/GSNET.git 
cd GSNET-main
```
3. Prepare dataset <br>

**MNIST** dataset can be downloaded automatically when you run the code <br>
**NORB** dataset can be downloaded online or in the following : <br>
https://pan.baidu.com/s/11Fhq7G2WMtoXM5a-9XA4vw passward: `d3xy`

4. Run experiment <br>

Each dataset has a trained .pt file in pts file, and run<br> 
`python GSNET.py` <br>
will show the result about **MNIST** dataset by default. <br>

if you want to run the entire training process by yourself, please select the 'train' model by using <br>
`python GSNET.py --mode train` <br>

5. Other usage <br>
 
use  `python GSNET.py -h`  for help <br>

## Results
![](/GSNET_MNIST.png)
