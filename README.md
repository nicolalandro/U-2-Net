# Train a U2Net

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