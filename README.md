# ChronoCell: AI-Powered Standard Cell Timing Characterization Platform

An end-to-end **ML for EDA (Machine Learning for Electronic Design Automation)** platform that predicts VLSI standard-cell timing characteristics across Process-Voltage-Temperature (PVT) corners using a neural network, replacing computationally expensive SPICE simulations with near-instant AI inference.

The platform characterizes digital standard cells, trains a machine learning surrogate model, visualizes timing behavior through an interactive web dashboard, and generates industry-standard Liberty (`.lib`) timing files.

---

## Overview

Traditional standard-cell characterization requires running thousands of SPICE simulations across multiple operating conditions, resulting in significant computational overhead.

ChronoCell accelerates this process by:

- Generating timing data across PVT corners
- Training a neural network to learn delay behavior
- Predicting propagation delay with high accuracy
- Providing real-time timing analysis through a web interface
- Exporting industry-standard Liberty timing libraries

This project demonstrates how machine learning can accelerate semiconductor design workflows while maintaining engineering-grade accuracy.

---

## Key Features

### AI-Based Timing Prediction
- Predicts propagation delay (`tpd`) for:
  - Inverter (INV)
  - NAND2 Gate
  - D Flip-Flop (DFF)

### PVT-Aware Characterization
Supports characterization across:

#### Process Corners
- TT (Typical-Typical)
- FF (Fast-Fast)
- SS (Slow-Slow)
- FS (Fast-Slow)
- SF (Slow-Fast)

#### Voltage Range
- 1.62V
- 1.80V
- 1.98V

#### Temperature Range
- -40°C to 125°C

### Machine Learning Pipeline
- Automated data generation
- Feature engineering
- Data preprocessing
- Neural network training
- Model evaluation
- Real-time inference

### Interactive Dashboard
- Live delay prediction
- Parameter sweep visualization
- PVT corner analysis
- Performance metrics dashboard
- Liberty file download

### Industry Output
- Generates Liberty (`.lib`) timing files
- Mimics real semiconductor characterization workflows

---

## Architecture

```text
                 +----------------------+
                 |   SPICE / Synthetic  |
                 |    Data Generation   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Data Processing &   |
                 | Feature Engineering  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   Neural Network     |
                 |      Training        |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Model Evaluation &   |
                 | Performance Analysis |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Real-Time Inference  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Flask Web Dashboard  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Liberty File Export  |
                 +----------------------+
```

---

## Technology Stack

### Languages
- Python
- HTML
- CSS
- JavaScript

### Machine Learning
- PyTorch
- Scikit-learn
- NumPy
- Pandas

### EDA & VLSI
- SPICE
- ngspice
- Liberty Format (.lib)

### Backend
- Flask
- REST APIs

### Visualization
- Chart.js
- Matplotlib

### Development Tools
- Git
- GitHub
- Jupyter Notebook

---

## Dataset Features

The model learns timing behavior using:

| Feature | Description |
|----------|-------------|
| VDD | Supply Voltage |
| Temperature | Operating Temperature |
| Load Capacitance | Output Load |
| Drive Strength | Cell Drive Capability |
| Process Corner | Manufacturing Variation |
| Cell Type | INV / NAND2 / DFF |

Engineered features include:

- VDD²
- 1 / Drive Strength
- VDD − Threshold Voltage
- Log-transformed delay targets

---

## Machine Learning Model

### Model Type
Multi-Layer Perceptron (MLP)

### Training Techniques
- AdamW Optimizer
- Huber Loss
- Early Stopping
- Learning Rate Scheduling
- Feature Standardization

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

### Performance

| Metric | Result |
|----------|----------|
| MAE | < 1 ps |
| MAPE | < 2% |
| R² Score | > 0.998 |
| Inference Speed | Microseconds |
| Speedup vs SPICE | Up to 500,000× |

---

## Project Structure

```text
ChronoCell-AI/
│
├── spice/
│   ├── models/
│   ├── inv_tb.sp
│   ├── nand2_tb.sp
│   ├── dff_tb.sp
│   └── run_sweep.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── liberty/
│   ├── process_data.py
│   └── gen_liberty.py
│
├── ml/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── checkpoints/
│
├── web/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── notebooks/
│
├── requirements.txt
├── run_all.py
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/ChronoCell-AI.git

cd ChronoCell-AI
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Complete Pipeline

```bash
python run_all.py
```

This performs:

1. Dataset Generation
2. Data Processing
3. Model Training
4. Model Evaluation
5. Liberty Generation

---

## Launch Dashboard

```bash
python web/app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Dashboard Features

### Interactive Timing Predictor
Predict delay instantly using:
- Voltage
- Temperature
- Load Capacitance
- Drive Strength
- Process Corner

### Parameter Sweep Visualizer
Analyze timing trends against:
- Voltage
- Temperature
- Load

### PVT Corner Table
Generate complete timing tables across:
- Process
- Voltage
- Temperature

### Model Accuracy Dashboard
Visualize:
- MAE
- MAPE
- R²
- Error Histograms
- Prediction Correlation

### Liberty Export
Download generated `.lib` timing libraries.

---

## Applications

- Standard Cell Characterization
- Timing Analysis
- Static Timing Analysis Research
- Semiconductor Design Automation
- ML for EDA Research
- Digital IC Design
- Timing Library Generation

---

## Future Enhancements

- OpenSTA Integration
- Additional Standard Cells
- Sky130 Full PDK Support
- Slew/Capacitance Lookup Tables
- Uncertainty Estimation
- FPGA Timing Prediction
- Graph Neural Network Models
- Cloud-Based Characterization Service

---

## Project Domain

**Primary Domain:** VLSI Design & Electronic Design Automation (EDA)

**Subdomains:**
- Machine Learning
- Semiconductor Engineering
- Digital IC Design
- Timing Analysis
- Design Automation
- ML for EDA

---

## Author

**Shreyank Bandi**

AI • Machine Learning • VLSI Design • Electronic Design Automation (EDA)

---
⭐ If you found this project useful, consider giving it a star.
