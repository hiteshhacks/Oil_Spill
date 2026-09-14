# 🌊 AI Multi-Modal Oil Spill Detection & Vessel Attribution System (SIH 2025)

An end-to-end prototype designed for **Smart India Hackathon 2025** that integrates **Sentinel-1 SAR Satellite Imagery**, **AIS Vessel Kinematics**, **Physics-Based Ocean Drift Modeling**, and **Explainable AI (XAI)** to detect offshore oil spills and attribute them to responsible suspect vessels.

---

## 🏗️ Architecture Overview

```text
[Ingestion]                 [Processing]               [Detect]                     [Vessel Attribution]          [Result]
AIS Data from Vessel  -->  Physics Drift Anomaly  -->  Satellite SAR Imagery  -->   Spill + Environmental   -->  Multi-Model Fusion
(Speed, Course, Time)      Detection (Low/High)        (Noise removal & U-Net)      Drift Backtracking Model     (AIS + SAR + Weather)
                                                                                                                   │
                                                                                                                   ▼
                                                                                                            Final Risk Score
                                                                                                           (Low / Med / High)
                                                                                                                   │
                                                                                                                   ▼
                                                                                                           Explainable AI (XAI)
```

---

## 📁 Project Structure

```text
oil_spill_detection/
│
├── data/
│   ├── ais/                     # Test SAR images & AIS records
│   └── satellite/               # Satellite SAR imagery (class_1_*.jpg)
│
├── src/
│   ├── ais/
│   │   ├── __init__.py
│   │   └── tracker.py           # Trajectory reconstruction & kinematic anomaly detector
│   │
│   ├── satellite/
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # Bilateral speckle filter & CLAHE contrast enhancement
│   │   └── detector.py          # Adaptive dark slick segmentation & morphological analysis
│   │
│   └── fusion/
│       ├── __init__.py
│       ├── drift.py             # Leeway ocean drift model & origin backtracking
│       └── attribution.py       # Multi-modal fusion, risk scoring & Explainable AI (XAI)
│
├── notebooks/                   # Exploratory analysis
├── app.py                       # Interactive Streamlit Web Dashboard
├── requirements.txt             # Project dependencies
└── README.md
```

---

## 🚀 How to Run the Prototype

1. **Activate your Python environment**
2. **Install requirements:**
   ```bash
   python -m pip install -r requirements.txt
   ```
3. **Launch the Streamlit Dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
4. Open your browser at `http://localhost:8501`.

---

## 🌟 Key Features

1. **🛰️ SAR Image Segmentation**:
   - Adaptive speckle noise reduction (Bilateral + Median filtering).
   - Dynamic dark-spot thresholding & CLAHE enhancement.
   - Extracts surface area ($km^2$), perimeter, slick elongation, and estimated volume (barrels).

2. **🚢 AIS Kinematic Anomaly Detection**:
   - Detects abnormal vessel deceleration (e.g. $14.5 \text{ kn} \rightarrow 3.1 \text{ kn}$) and sharp loitering turns inside discharge corridors.

3. **🌊 Physics-Based Drift Model**:
   - Combines ocean currents and 3% wind leeway with Coriolis deflection to backtrack release origin and project 48-hour forward spread.

4. **🗺️ Interactive Folium Marine GIS**:
   - Displays live satellite slicks, vessel routes, suspect event flags, and drift vectors on interactive nautical dark maps.

5. **🧠 Explainable AI (XAI) Incident Audit**:
   - Generates human-readable evidence points explaining *why* a vessel was flagged as High Risk, complete with multi-dimensional radar charts and JSON export.
