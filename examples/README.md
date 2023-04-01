# Example Scripts

This directory contains some example scripts.

**Iris Classification**

```shell
pip install sklearn
PYTHONPATH=./ python3 examples/test_einet_iris.py --device cpu --lr 0.07 -D 1 --epochs 40
```

**Iris Classification with Class Conditionals**

```shell
PYTHONPATH=./ python3 examples/test_ccleinet_iris.py  --device cpu --lr 0.4 --epochs 40
```
