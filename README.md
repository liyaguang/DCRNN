# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.


## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- tensorflow>=1.3.0
- pyaml


Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/` folder.
Besides, the locations of sensors Los Angeles are available at [data/sensor_graph/graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv).
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```
The generated train/val/test dataset will be saved at `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.


## Run the Pre-trained Model on METR-LA

```bash
# METR-LA
python run_demo.py --config_filename=data/model/pretrained/METR-LA/config.yaml

# PEMS-BAY
python run_demo.py --config_filename=data/model/pretrained/PEMS-BAY/config.yaml
```
The generated prediction of DCRNN is in `data/results/dcrnn_predictions`.


## Model Training
```bash
# METR-LA
python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-BAY
python dcrnn_train.py --config_filename=data/model/dcrnn_bay.yaml
```
Each epoch takes about 5min or 10 min on a single GTX 1080 Ti for METR-LA or PEMS-BAY respectively. 

There is a chance that the training loss will explode, the temporary workaround is to restart from the last saved model before the explosion, or to decrease the learning rate earlier in the learning rate schedule. 

## Graph Construction
 As the currently implementation is based on pre-calculated road network distances between sensors, it currently only
 supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`).

```bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

More details are being added ...

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
