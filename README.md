# Enhancing Health Federated Learning Security Against Poisoning Attacks with Blockchain and AWB-FedAvg

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project proposes a **Secure Healthcare Federated Learning Framework** that integrates:
- **Multi-metric anomaly detection** for identifying poisoning attacks
- **Adaptive Weighted Byzantine-resilient FedAvg (AWB-FedAvg)** aggregation algorithm
- **Blockchain-based verification** and auditability of model updates

The system is designed to enable privacy-preserving collaborative machine learning across healthcare institutions (hospitals, diagnostic centers) while defending against Byzantine and poisoning attacks.

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Hospital 1  │    │  Hospital 2  │    │  Hospital N  │
│  (Client)    │    │  (Client)    │    │  (Client)    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       │    Local Model Updates (Encrypted)    │
       └───────────┬───────┴───────────────────┘
                   │
         ┌─────────▼─────────┐
         │   AWB-FedAvg       │
         │   Aggregation      │
         │   Server           │
         │                    │
         │  • Anomaly Det.    │
         │  • Trust Scoring   │
         │  • Weighted Avg.   │
         └─────────┬──────────┘
                   │
         ┌─────────▼─────────┐
         │   Blockchain       │
         │   Network          │
         │                    │
         │  • Model Hashes    │
         │  • Reputation      │
         │  • Audit Trail     │
         └────────────────────┘
```

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Federated Learning** | Privacy-preserving distributed training across hospitals |
| **AWB-FedAvg** | Adaptive Weighted Byzantine-resilient Federated Averaging |
| **Blockchain Integration** | Immutable audit trail and decentralized trust management |
| **Anomaly Detection** | Multi-metric detection of poisoning attacks |
| **Healthcare Analytics** | Comprehensive clinical performance metrics & visualizations |
| **Attack Simulation** | Random, sign-flip, and targeted attack models |

## 📊 Visualizations Generated

The system produces comprehensive analytics dashboards:

1. **Advanced Healthcare Dashboard** — 12-panel analytics with 3D performance surfaces, radar charts, network topology, and more
2. **Interactive Timeline** — Training timeline with performance events
3. **Comparative Analysis** — Security-performance trade-offs, clinical impact assessment
4. **Training Animation** — Animated GIF of the training process
5. **System Health Monitoring** — Convergence analysis, resource utilization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/Kommindala-Ramoji/Healthcare-Federated-Learning-AWB-FedAvg.git
cd Healthcare-Federated-Learning-AWB-FedAvg

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

```bash
python awb_fedavg_healthcare.py
```

Or open and run the Jupyter notebook:

```bash
jupyter notebook AWB_FedAvg_Healthcare.ipynb
```

## ⚙️ Configuration

You can customize the simulation via the `config` dictionary:

```python
config = {
    'num_clients': 8,          # Number of hospitals
    'num_malicious': 2,        # Number of malicious clients
    'num_rounds': 10,          # Training rounds
    'local_epochs': 3,         # Local training epochs per round
    'input_dim': 20,           # Input feature dimensions
    'hidden_dims': [32, 16],   # Hidden layer sizes
    'output_dim': 2,           # Output classes (healthy/diseased)
    'num_samples': 2000,       # Total patient records
    'non_iid': True,           # Non-IID data distribution
    'disease_prevalence': 0.25,# Disease prevalence rate
    'hospital_type': 'mixed'   # Hospital specialization type
}
```

## 📁 Project Structure

```
├── awb_fedavg_healthcare.py       # Main implementation
├── AWB_FedAvg_Healthcare.ipynb    # Jupyter notebook version
├── requirements.txt               # Python dependencies
├── docs/
│   └── Project_Report.docx        # Detailed academic project report
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## 📈 Results

The AWB-FedAvg algorithm demonstrates:
- **High accuracy** in medical disease prediction tasks
- **Effective attack detection** against Byzantine and poisoning attacks
- **Adaptive trust management** that isolates malicious participants
- **Blockchain-verified** model integrity and auditability

## 👥 Team

| Roll Number | Name |
|-------------|------|
| 22F61A05B4 | **Kommindala Ramoji** |
| 22F61A0564 | Muntha Keerthi |
| 23F65A0510 | Tharigonda Reddy Lakshmi |
| 22F61A0571 | R Lathish |

**Guide:** Dr. J. Sridhar, M.E., Ph.D., Associate Professor, Department of CSE  
**Institution:** Siddharth Institute of Engineering & Technology (Autonomous), Puttur, A.P.  
**Program:** B.Tech in Computer Science and Engineering (IV Year, II Semester) — 2026

## 📚 References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Blockchain-based Federated Learning frameworks for healthcare security
- Byzantine fault tolerance in distributed machine learning systems

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> **Note:** This project was developed as part of the B.Tech academic curriculum at SIETK. The dataset used is synthetically generated for demonstration purposes.
