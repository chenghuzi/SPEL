# SPEL: Sequence Prediction Error Learning

Source code for [Monkey Prefrontal Cortex Learns to Minimize Sequence Prediction Error](https://www.biorxiv.org/content/10.1101/2024.02.28.582611)

**System requirements**
OS: Red Hat Enterprise Linux Server 7.6 (Maipo)
Python: 3.10
Other 3rd party libs: see requirements.txt
CUDA: 11.7

## Quickstart

Install dependencies:
```
pip install -r requirements.txt
```
> The installation may take minutes to hours depending on the hardware and your network conditions.

Run the code:

```
python run_train.py # run simulation
```
Perform analysis and plot figure
```
python run_analysis.py --seed 12 --sim-dir /path/to/your/stored/data
```
> The installation may take a few hours depending on the hardware conditions.

