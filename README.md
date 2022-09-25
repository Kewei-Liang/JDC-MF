<p>
  This is the python implementation of the paper "The Joint Detection and Classification model for spatiotemporal action localization of primates in a group" and our paper will be published soon.
</p>

<details open>
<summary>Install</summary>
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
cd JDC-MF
pip install -r requirements.txt  # install
```
  
</details>
  
<details open>
<summary>Train with train.py</summary>

`detect.py` runs train and saving results to `runs/train`.

```bash
python train.py 
```

</details>

</details>
  
<details open>
<summary>Validation with val.py</summary>

`detect.py` runs validation and saving results to `runs/val`.

```bash
python val.py 
```

</details>

</details>
  
<details open>
<summary>Inference with detect.py</summary>

`detect.py` runs inference and saving results to `runs/detect`.

```bash
python detect.py 
```

</details>

