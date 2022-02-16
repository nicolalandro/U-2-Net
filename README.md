# Train a U2Net

* libs
```
pip install pytorch-ignite
```

* test
```
cd saved_models
mkdir u2net
cd u2net
wget https://github.com/nicolalandro/U-2-Net/releases/download/0.1/u2net.pth
cd ..
cd ..
python3.7 u2net_test.py
```

* train
```
python3.7 u2net_train.py
```

* train daedalus
```
mkdir saved_models/u2net_daedalus1
CUDA_VISIBLE_DEVICES="1" nohup python3.7 u2net_train_daedalus.py > daedalus.log  2>&1 &
```
* predict
```
CUDA_VISIBLE_DEVICES=0 python3.7 u2net_test_daedalus.py
# results in test_data/u2net_results/<nome_data>.png
```
* test metrics
```
CUDA_VISIBLE_DEVICES="0,1" nohup python3.7 u2net_test_daedalus_metrics.py > test_daedalus2.log 2>&1 &
```