# EV Charging & Battery Intelligence Dataset Generator

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-MultiTable%20Relational-blue?style=for-the-badge" alt="Dataset Type">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Kaggle-Ready-orange?style=for-the-badge" alt="Kaggle Ready">
</p>

A reproducible Python pipeline for generating a comprehensive, multi-table synthetic **Electric Vehicle (EV) Charging & Battery Intelligence Ecosystem**. Built to support advanced machine learning research, predictive analytics, energy demand forecasting, and anomaly detection.

- **Author:** Mobeen Fatima (`MobeenFatimaa`)
- **Target Platforms:** Kaggle & GitHub

---

## Relational Data Architecture

Rather than a single flat file, this dataset is structured as a relational database connected via explicit primary and foreign keys (`vehicle_id`, `station_id`, `session_id`).

```text
                         +----------------------+
                         |   ev_vehicles.csv    |
                         | (10k rows / 1 per EV) |
                         +----------+-----------+
                                    | 1
                                    | N
+--------------------+    +---------+------------------+    +-----------------------+
| ev_stations.csv    | 1  | ev_charging_sessions.csv  | 1  | ev_battery_health.csv |
| (500 rows / 1 per  +----+ (250k rows / 1 per session)+----+ (100k rows / periodic |
| charging station)  | N  +----------+------------------+ N  | health snapshots)     |
+---------+----------+               | 1                   +-----------------------+
          |                          |
          | 1                        | N
          |                     +----+-------------------------+
          |                     | ev_charging_telemetry.csv    |
          +---------------------+ (~1.04M rows / 5-min intervals)
          |                     +------------------------------+
          |
          | 1
          | N
+---------+------------------------------+
| ev_station_hourly_demand.csv           |
| (250,000 station-hour rollups)         |
+----------------------------------------+
```

## Dataset Components & Schema

| File Name | Row Count | Observational Unit | Key Attributes |
|---|---:|---|---|
| `ev_charging_sessions.csv` | 250,000 | Per charging session | Start/end times, SOC changes, energy delivered, pricing, costs, and anomaly flags. |
| `ev_vehicles.csv` | 10,000 | Per electric vehicle | Battery capacities (nominal/usable), chemistries (LFP/NMC/NCA), range, and charging limits. |
| `ev_charging_stations.csv` | 500 | Per charging hub | Hub types (Workplace/Highway/Public), geo-coordinates, power limits, and solar capacity. |
| `ev_battery_health.csv` | 100,000 | Periodic vehicle snapshot | Degradation tracking, capacity fade percentage, equivalent full cycles, and risk categories. |
| `ev_charging_telemetry.csv` | ~1,039,402 | 5-minute telemetry step | Time-series sensor logs recording power curves and thermal responses during sessions. |
| `ev_station_hourly_demand.csv` | 250,000 | Per station-hour | Aggregated load, active session counts, utilization percentages, and peak indicators. |

---

## Key Machine Learning Tasks Supported

1. **Energy Delivery Regression:** Predict `energy_delivered_kwh` using vehicle specifications, initial SOC, and environmental conditions.
2. **Battery SOH Prediction:** Estimate battery health fade over time using thermal exposure and cumulative throughput.
3. **Station Demand Forecasting:** Forecast `next_hour_energy_demand_kwh` using time-series lags and weather features.
4. **Session Anomaly Detection:** Identify unusual charging interruptions or thermal spikes using anomaly flags.
5. **Cost Optimization:** Study time-of-use tariffs and session fees to optimize vehicle charging behavior.

---

## Quick Start & Installation

### Prerequisites

Ensure you have Python 3.8+ installed. The pipeline relies on vectorized operations via Pandas and NumPy for high-performance generation.

### 1. Clone the Repository

```bash
git clone https://github.com/mobeenfatimah/ev-intelligence-dataset-generator.git
cd ev-intelligence-dataset-generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset

Run the main generation script to build all 6 relational tables inside an `./ev_dataset/` directory:

```bash
python generate.py
```

### 4. Run Automated Validation

Verify relational integrity, physical laws (e.g., efficiency bounds, SOC logic), and data boundaries:

```bash
python validate_ev_dataset.py
```

---

## Baseline Machine Learning Model (`train_baseline.py`)

To demonstrate how practitioners can leverage this relational dataset for predictive modeling, the repository includes a baseline machine learning script (`train_baseline.py`) that predicts energy delivery without data leakage.

### Baseline Performance (LightGBM Regressor)

Evaluated on a chronological test split of 37,500 charging sessions:

- **R² Score:** `0.8653` (explains 86.5% of variance)
- **Mean Absolute Error (MAE):** `4.1944 kWh`
- **Root Mean Squared Error (RMSE):** `5.0100 kWh`

### Top Predictive Features

1. `soc_start_pct` (Initial State of Charge)
2. `usable_battery_capacity_kwh`
3. `battery_capacity_kwh`
4. `ambient_temperature_c`
5. `rated_range_km`

### Run Baseline Training

```bash
python train_baseline.py
```

---

## Data Generation & Physical Integrity Rules

- **Energy Conservation Law:** Grid energy drawn strictly scales above battery energy delivered based on thermal efficiency.
- **Tapering Power Curves:** DC fast-charging telemetry curves realistically decay as battery state-of-charge approaches 80%+.
- **Referential Consistency:** Zero foreign key orphaned records across all relational joins.

---

## Contact & Connect

- **GitHub:** [MobeenFatimaa](https://github.com/MobeenFatimaa)
- **LinkedIn:** [Mobeen Fatima](https://www.linkedin.com/in/mobeen-fatima-599a35347/)
- **Kaggle:** [mobeenfatimah](https://www.kaggle.com/mobeenfatimah)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.
