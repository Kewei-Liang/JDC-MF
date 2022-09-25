
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://raw.githubusercontent.com/Kewei-Liang/JDC-MF/main/Figures/test.jpg"></a>
</p>
<br>

## <div align="center">Introduction</div>
<p>
  This is the python implementation of the paper "The Joint Detection and Classification model for spatiotemporal action localization of primates in a group" and our paper will be published soon.
</p>

## <div align="center">Quick Start a Example</div>
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
<summary>Data preparation</summary>
A example of unzip data
```bash
cd data_example
unzip frames
unzip labels
```

</details>

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

