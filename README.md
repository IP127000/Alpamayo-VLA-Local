# Alpamayo-1-Local
VLA model, a local, offline‑running adaptation of NVIDIA’s Alpamayo 1. Run models, process data and visualize results entirely on your own machine.

## Visualization 

| paras             | values                                 |
|-------------------|----------------------------------------|
| clip‑id           | eed514a0‑a366‑4550‑b9bd‑4c296c531511   |
| t0‑us             | 10000000                               |

| result            | values                                 |
|-------------------|----------------------------------------|
| Chain‑of‑thought  | Adapt speed for the left curve ahead   |
| minADE            | 1.8058 m                               |

<img src="images/result_alpamayo.webp" width="70%" alt="alpamayo result">

## Usage

```bash
git clone https://github.com/IP127000/Alpamayo-VLA-Local.git
```
```bash
cd Alpamayo-VLA-Local
```
```bash
pip install -r requirements.txt
```
```bash
python inference.py
```
