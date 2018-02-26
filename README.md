# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This is a TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926).


## Requirements
- hyperopt>=0.1
- scipy>=0.19.0
- numpy>=1.12.1
- pandas==0.19.2
- tensorflow>=1.3.0
- python 2.7

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Traffic Data
The traffic data file for Los Angeles is available [here](https://drive.google.com/open?id=1tjf5aXCgUoimvADyxKqb-YUlxP8O46pb), and should be
put into the `data/` folder.


## Graph Construction
 As the currently implementation is based on pre-calculated road network distances between sensors, it currently only
 supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`).

```bash
python gen_adj_mx.py  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

## Train the Model
```bash
python dcrnn_seq2seq_train.py --config_filename=data/model/dcrnn_config.json
```


## Run the Pre-trained Model

```bash
python run_demo.py
```
The generated prediction of DCRNN is in `data/results/dcrnn_predictions_[1-12].h5`.


More details are being added ...

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{li2017dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```
