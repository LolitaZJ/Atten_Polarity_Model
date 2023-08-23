Usage

Date: 2023.08.23

Ji Zhang

# Atten_PD_Model 
Atten_PD_Model is an attention-based neural network to achieve the polarity determination.

Here, we use 4 datasets to train our model and obtain 4 trained models in 4 areas.

We alse use two model structures (PD_model and Ross_model)

**Well trained models are open source!**
 
We will then open source a more general model or model structures.   

If you are interested in our work, welcome to join us!

## 1. Set envirnoment 
> tensorflow >= 2.0  
> python >= 3.8  
> keras-self-attention   
     `pip install keras-self-attention`

## 2. Network Architecture
![Network Architecture](./Network_Structures.png)

## 3. Codes
- Attention_PD_MASTER_V01.py    
Main code 

- APP_PD_model.py  
Build model

- APP_PD_utils.py   
Functions of generators and plots

## 4. Parameters
-- model_name:  bulid_pd_model/cnn_ross/bulid_pp_model/pd_model

> **bulid_pd_model**: attention-based polarity determination   
> **cnn_ross**: CNN model based Ross et al paper  
> **bulid_pp_model**: attention-based phase picking & polarity determination   
> **pd_model**: attention-based polarity determination (2D)  

-- save_name: the model name (pd_model_ins)

-- dataset: Four datasets (ins/hinet/scsn/pnw)

- [ins](https://www.pi.ingv.it/banche-dati/instance/) or https://www.pi.ingv.it/banche-dati/instance/
- [hinet](https://www.hinet.bosai.go.jp/) or https://www.hinet.bosai.go.jp/
- [pnw](https://github.com/niyiyu/PNW-ML/tree/main) or https://github.com/niyiyu/PNW-ML/tree/main
- [scsn](https://scedc.caltech.edu/data/deeplearning.html#picking_polarity) or https://scedc.caltech.edu/data/deeplearning.html#picking_polarity

-- epochs: number of epochs (default: 500)

-- batch_size: default=1024
                        ,
-- learning_rate: default=0.001,

-- patience: early stopping  (default=20)

-- monitor: monitor the val_loss/loss/acc/val_acc  (default="val_accuracy")

-- monitor_mode: min/max/auto (default="max")

-- loss: loss fucntion  (default='categorical_crossentropy')


## 5. Fast run!

### a) INSTANCE dataset
- PD_model

>  `python APP_PD_MASTER.py --save_name=pd_model_ins  --dataset=ins --GPU=3`

- Ross_model

>  `python APP_PD_MASTER.py --save_name=ross_model_ins  --model_name=cnn_ross --dataset=ins --GPU=3`

### b) Hi-Net dataset
- PD_model

`python APP_PD_MASTER.py --save_name=pd_model_hinet --dataset=hinet --GPU=2`

- Ross_model

>  `python APP_PD_MASTER.py --save_name=ross_model_hinet  --model_name=cnn_ross --dataset=ins --GPU=2`

### c) SCSN dataset
- PD_model

`python APP_PD_MASTER.py --save_name=pd_model_scsn --dataset=scsn --GPU=1`

- Ross_model

>  `python APP_PD_MASTER.py --save_name=ross_model_scsn  --model_name=cnn_ross --dataset=ins --GPU=1`

### d) PNW dataset
- PD_model

`python APP_PD_MASTER.py --save_name=pd_model_pnw --dataset=pnw -GPU=0`

- Ross_model

>  `python APP_PD_MASTER.py --save_name=ross_model_pnw  --model_name=cnn_ross --dataset=ins --GPU=0`

## 6. Model
We save the models in `./model/` file. 
You can use them directly!!!

### PD_model
- pd_model_ins.h5
- pd_model_hinet.h5
- pd_model_scsn.h5
- pd_model_pnw.h5
### Ross_model
- ross_model_ins.h5
- ross_model_hinet.h5
- ross_model_scsn.h5
- ross_model_pnw.h5

## Related papers:
- Zhang, J., Li, Z., & Zhang, J. (2023). Simultaneous Seismic Phase Picking and Polarity Determination with an Attention‐Based Neural Network. Seismological Society of America, 94(2A), 813-828.
- Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., and Lauciani, V., INSTANCE – the Italian seismic dataset for machine learning, Earth Syst. Sci. Data, 13 (12), 5509 – 5544, doi:10.5194/essd-13-5509-2021.
- Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., & Wright, A. (2023). Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368.
- Ross, Z. E., Meier, M. A., & Hauksson, E. (2018). P wave arrival picking and first‐motion polarity determination with deep learning. Journal of Geophysical Research: Solid Earth, 123(6), 5120-5129.
- K.Obara, K.Kasahara, S.Hori and Y.Okada, 2005, A densely distributed high-sensitivity seismograph network in Japan: Hi-net by National Research Institute for Earth Science and Disaster Prevention, Review of Scientific Instruments, 76, 021301, doi:10.1063/1.1854197.
