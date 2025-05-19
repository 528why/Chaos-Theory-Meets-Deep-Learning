# Chaotic Systems Enhanced Deep Learning for Time Series Forecasting

This repository presents a novel approach that integrates **chaotic systems** with **deep learning** to improve time series forecasting performance. Specifically, we introduce a deep learning framework based on the **Chen chaotic system**, which harnesses the inherent randomness, sensitivity, and diversity of chaos to boost model accuracy and efficiency.

## 🔍 Overview

- 💡 Combines chaos theory with deep learning to enhance forecasting.
- 🌀 Uses the **Chen system** for chaotic initialization and dynamic behavior.
- ⚙️ Benchmarks include:
  - LSTM (Long Short-Term Memory)
  - N-BEATS (Neural Basis Expansion Analysis)
  - Informer
- 🔁 Each baseline is compared with its **chaotic-enhanced counterpart**.

## 📊 Experiments

- **Datasets**: 13 publicly available time series datasets (stock prices, electricity usage, air quality, etc.)
- **Metrics**:
  - Forecasting accuracy
  - Runtime
  - Resource usage
- Results show **consistent improvements** using chaotic systems across all benchmarks.

## 📥 Dataset Download

You can download the datasets used in our experiments from the following sources:

- **ETT** (Electricity Transformer Temperature):  
  [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

- **ECL** (Electricity Load Diagrams):  
  [https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)

- **Weather (NOAA)**:  
  [https://www.ncei.noaa.gov/data/local-climatological-data/](https://www.ncei.noaa.gov/data/local-climatological-data/)

- **GEI** (Shenzhen Stock Exchange Index):  
  [https://www.szse.cn/certificate/secondb/](https://www.szse.cn/certificate/secondb/)

- **HSI** (Hang Seng Index):  
  [https://www.hsi.com.hk/schi](https://www.hsi.com.hk/schi)

- **S&P50**:  
  [https://www.spglobal.com/spdji/en/indices/equity/sp-500](https://www.spglobal.com/spdji/en/indices/equity/sp-500)

## 🚀 Key Contributions

- ✅ A generalizable framework for integrating chaos with any time series model.
- ✅ Demonstrated benefits of chaotic mapping in training initialization and model dynamics.
- ✅ Extensive experiments proving effectiveness across multiple domains.

## 📁 Coming Soon

- [ ] More chaotic-enhanced network architectures
<!-- - [ ] Code for model training & evaluation
- [ ] Reproducible experiment instructions -->

## 📜 Citation

If you find this work useful, please consider citing our paper:

@article{jia2024chaos,
  title={Chaos theory meets deep learning: A new approach to time series forecasting},
  author={Jia, Bowen and Wu, Huyu and Guo, Kaiyu},
  journal={Expert Systems with Applications},
  volume={255},
  pages={124533},
  year={2024},
  publisher={Elsevier}
}

