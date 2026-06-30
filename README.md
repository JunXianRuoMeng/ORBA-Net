# Multi‑Modal Fusion for Cyber‑Physical Intrusion Detection

This repository provides an experimental pipeline for **multi‑modal intrusion detection** that fuses network traffic and host‑process statistics. It is designed to work with two publicly available benchmark datasets (see below) and implements a temporal‑alignment fusion strategy to detect attacks under both cyber and physical scenarios.

---

## 🔔 Important Note

Both datasets are **large**. **They are NOT included in this repository.** You must download them separately from the official sources and place them into the corresponding directories as described in the setup section.

---

## 📊 Datasets

### 1. WDT – Hardware‑in‑the‑Loop Water Distribution Testbed

- **Paper:** *"A Hardware‑in‑the‑Loop Water Distribution Testbed Dataset for Cyber‑Physical Security Testing,"*  
- **Content:** Physical process measurements (pressure, tank levels, valve/pump states) and MODBUS TCP/IP network traffic with 28 attack scenarios (MITM, DoS, scanning, physical leaks, sensor/pump failures).  
- **Download:** [IEEE DataPort](https://ieee-dataport.org/open-access/hardware-loop-water-distribution-testbed-wdt-dataset-cyber-physical-security-testing) *(IEEE account required)*

### 2. CREME – Comprehensive Real‑world Multi‑source Enterprise Dataset

- **Paper:** *"CREME: A toolchain of automatic dataset collection for machine learning in intrusion detection"*).  
- **Content:** Large‑scale enterprise telemetry including NetFlow/IPFIX, host process stats (CPU, memory, I/O), logs, and multi‑class attack labels.  
- **Download:** [Google Drive](https://drive.google.com/drive/folders/1DNXFtMRrFUkir4sW2cmyCR25GjEmBMEs) *(choose relevant splits)*

---

## 🖥️ Environment

Dependencies are provided in two formats:

- **pip** – `requirements.txt`  
- **conda** – `environment.yml`  

Use either to set up your environment:

```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml



🚀 Quick Start
Download datasets and place them under src/CREME/ and sre/WDT/.

Install dependencies (see above).

Run aggregation for each dataset:

bash
python src/CREME/Data_Fusion.py
python src/WDT/Data_Fusion.py


cd src/WDT/ORBA-Net
python ORBA-Net.py
cd ../..

cd src/CREME/ORBA-Net
python ORBA-Net.py
cd ../..


