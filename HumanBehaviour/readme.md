# Human Behaviour classification
## You have to adjust your own threshold according to your enviroment.
#### Note: This model was for anomaly detection in whole video we still implemented it in real-time with some trade-offs.
### Implemented with spatial diffrence Learning.(New)
![Paper](1701.01546.pdf)
## Dataset
### I Used the Avenue Datset.
### Processing
#### 1) Took every 10 frames.
#### 2) Grayscaled and resized to 227x227

## Model Used
### I created a custom resnet model using pytorch
![Model](images/model.png)
### LSTM Layer
![lstm](images/lstm.png)
### Training
#### ACC was 73% with 30 epochs.
![train](images/train.png)
## Run on your own Desktop
```
pip install -r requirements.txt
python3 frame_prediction_lstm_behaviour.py
```
#### Open .ipynb notebook to train on your on dataset.
#   W e a p o n - D e t e c t i o n - a n d - V i o l e n c e - A n a l y s i s  
 