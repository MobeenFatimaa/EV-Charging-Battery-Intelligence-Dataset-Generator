import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_full_ev_dataset(output_dir="./ev_dataset", seed=42):
    """
    Generates all 6 relational tables for the EV Charging & Battery Intelligence Dataset:
    1. ev_vehicles.csv (10,000 rows)
    2. ev_charging_stations.csv (500 rows)
    3. ev_charging_sessions.csv (250,000 rows)
    4. ev_battery_health.csv (100,000 rows)
    5. ev_charging_telemetry.csv (~1,500,000 rows)
    6. ev_station_hourly_demand.csv (250,000 rows)
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print("Generating EV Charging & Battery Intelligence Dataset...")

    # -------------------------------------------------------------
    # FILE 1: ev_vehicles.csv (10,000 rows)
    # -------------------------------------------------------------
    print("1/6 Generating ev_vehicles.csv...")
    num_vehicles = 10000
    vehicle_ids = [f"EV_{i:05d}" for i in range(1, num_vehicles + 1)]
    categories = ['Sedan', 'SUV', 'Hatchback', 'Van']
    chemistries = ['LFP', 'NMC', 'NCA']
    
    cat_specs = {
        'Sedan': {'cap_mean': 65, 'cap_std': 8, 'eff': 15.5},
        'SUV': {'cap_mean': 88, 'cap_std': 12, 'eff': 21.0},
        'Hatchback': {'cap_mean': 48, 'cap_std': 6, 'eff': 13.8},
        'Van': {'cap_mean': 98, 'cap_std': 10, 'eff': 25.2}
    }
    
    assigned_cats = np.random.choice(categories, size=num_vehicles, p=[0.40, 0.35, 0.20, 0.05])
    nom_caps = np.array([
        np.clip(np.random.normal(cat_specs[c]['cap_mean'], cat_specs[c]['cap_std']), 30, 130)
        for c in assigned_cats
    ])
    usable_caps = nom_caps * np.random.uniform(0.92, 0.96, size=num_vehicles)
    
    df_vehicles = pd.DataFrame({
        'vehicle_id': vehicle_ids,
        'vehicle_model': [f"Model-{c[0]}{i%20 + 1}" for i, c in enumerate(assigned_cats)],
        'vehicle_category': assigned_cats,
        'vehicle_year': np.random.choice([2021, 2022, 2023, 2024, 2025], size=num_vehicles, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
        'battery_capacity_kwh': np.round(nom_caps, 2),
        'usable_battery_capacity_kwh': np.round(usable_caps, 2),
        'battery_chemistry': np.random.choice(chemistries, size=num_vehicles, p=[0.35, 0.50, 0.15]),
        'rated_range_km': np.round(nom_caps / [cat_specs[c]['eff']/100 for c in assigned_cats], 1),
        'vehicle_efficiency_kwh_per_100km': [cat_specs[c]['eff'] for c in assigned_cats],
        'maximum_ac_charging_power_kw': np.random.choice([7.2, 11.0, 22.0], size=num_vehicles, p=[0.3, 0.6, 0.1]),
        'maximum_dc_charging_power_kw': np.random.choice([50.0, 150.0, 250.0, 350.0], size=num_vehicles, p=[0.2, 0.5, 0.2, 0.1]),
        'initial_battery_soh_pct': np.round(np.random.uniform(97.0, 100.0, size=num_vehicles), 2)
    })
    df_vehicles.to_csv(os.path.join(output_dir, "ev_vehicles.csv"), index=False)

    # -------------------------------------------------------------
    # FILE 2: ev_charging_stations.csv (500 rows)
    # -------------------------------------------------------------
    print("2/6 Generating ev_charging_stations.csv...")
    num_stations = 500
    station_ids = [f"STN_{i:04d}" for i in range(1, num_stations + 1)]
    station_types = ['Residential', 'Workplace', 'Public', 'Highway', 'Commercial']
    
    df_stations = pd.DataFrame({
        'station_id': station_ids,
        'station_name': [f"Hub-{i:03d} ChargeCenter" for i in range(1, num_stations + 1)],
        'station_type': np.random.choice(station_types, size=num_stations, p=[0.25, 0.25, 0.25, 0.15, 0.10]),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=num_stations),
        'latitude': np.round(np.random.uniform(34.0, 42.0, size=num_stations), 5),
        'longitude': np.round(np.random.uniform(-118.0, -73.0, size=num_stations), 5),
        'number_of_chargers': np.random.randint(2, 24, size=num_stations),
        'maximum_station_power_kw': np.random.choice([50, 150, 300, 600, 1200], size=num_stations),
        'renewable_energy_available': np.random.choice([True, False], size=num_stations, p=[0.4, 0.6]),
        'solar_capacity_kw': np.round(np.random.choice([0, 25, 50, 100], size=num_stations, p=[0.6, 0.2, 0.1, 0.1]), 1)
    })
    df_stations.to_csv(os.path.join(output_dir, "ev_charging_stations.csv"), index=False)

    # -------------------------------------------------------------
    # FILE 3: ev_charging_sessions.csv (250,000 rows)
    # -------------------------------------------------------------
    print("3/6 Generating ev_charging_sessions.csv...")
    num_sessions = 250000
    session_ids = [f"SES_{i:07d}" for i in range(1, num_sessions + 1)]
    
    v_indices = np.random.choice(num_vehicles, size=num_sessions)
    s_indices = np.random.choice(num_stations, size=num_sessions)
    
    selected_v = df_vehicles.iloc[v_indices].reset_index(drop=True)
    selected_s = df_stations.iloc[s_indices].reset_index(drop=True)
    
    start_date = datetime(2025, 1, 1)
    random_seconds = np.random.randint(0, 365 * 24 * 3600, size=num_sessions)
    start_times = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
    
    soc_start = np.random.uniform(10.0, 45.0, size=num_sessions)
    soc_target = np.random.uniform(75.0, 100.0, size=num_sessions)
    soc_change = soc_target - soc_start
    
    usable_cap = selected_v['usable_battery_capacity_kwh'].values
    energy_delivered_battery = usable_cap * (soc_change / 100.0)
    
    # Bound ambient temperature to prevent unbounded Gaussian outliers
    amb_temp = np.clip(np.random.normal(20.0, 10.0, size=num_sessions), -20.0, 50.0)
    charging_eff = np.where(amb_temp < 5.0, np.random.uniform(82.0, 88.0, size=num_sessions), np.random.uniform(90.0, 95.0, size=num_sessions))
    energy_drawn_grid = energy_delivered_battery / (charging_eff / 100.0)
    
    is_fast = selected_s['station_type'].isin(['Highway', 'Public']).values
    charger_limits = np.where(is_fast, selected_v['maximum_dc_charging_power_kw'].values, selected_v['maximum_ac_charging_power_kw'].values)
    max_supported_power = np.minimum(charger_limits, selected_s['maximum_station_power_kw'].values)
    avg_power_kw = max_supported_power * np.random.uniform(0.60, 0.85, size=num_sessions)
    
    duration_hours = energy_drawn_grid / avg_power_kw
    duration_minutes = duration_hours * 60.0
    end_times = [st + timedelta(minutes=float(dm)) for st, dm in zip(start_times, duration_minutes)]
    
    tariff = np.random.uniform(0.12, 0.45, size=num_sessions)
    energy_cost = energy_drawn_grid * tariff
    total_cost = energy_cost + np.random.choice([0.0, 1.5, 2.5], size=num_sessions)
    
    battery_soh = np.clip(100.0 - (np.array(random_seconds) / (365 * 24 * 3600)) * np.random.uniform(1.5, 4.0, size=num_sessions), 80.0, 100.0)
    anomaly_flag = np.random.choice([0, 1], size=num_sessions, p=[0.97, 0.03])
    
    df_sessions = pd.DataFrame({
        'session_id': session_ids,
        'vehicle_id': selected_v['vehicle_id'].values,
        'station_id': selected_s['station_id'].values,
        'session_start_time': start_times,
        'session_end_time': end_times,
        'session_duration_minutes': np.round(duration_minutes, 2),
        'soc_start_pct': np.round(soc_start, 2),
        'soc_end_pct': np.round(soc_target, 2),
        'soc_change_pct': np.round(soc_change, 2),
        'battery_soh_pct': np.round(battery_soh, 2),
        'energy_delivered_kwh': np.round(energy_delivered_battery, 2),
        'energy_drawn_from_grid_kwh': np.round(energy_drawn_grid, 2),
        'charging_efficiency_pct': np.round(charging_eff, 2),
        'average_charging_power_kw': np.round(avg_power_kw, 2),
        'ambient_temperature_c': np.round(amb_temp, 1),
        'electricity_price_per_kwh': np.round(tariff, 3),
        'total_session_cost': np.round(total_cost, 2),
        'charging_anomaly_flag': anomaly_flag
    })
    df_sessions.to_csv(os.path.join(output_dir, "ev_charging_sessions.csv"), index=False)
    # -------------------------------------------------------------
    # FILE 4: ev_battery_health.csv (100,000 rows)
    # -------------------------------------------------------------
    print("4/6 Generating ev_battery_health.csv...")
    num_health = 100000
    health_ids = [f"HLT_{i:07d}" for i in range(1, num_health + 1)]
    v_health_indices = np.random.choice(num_vehicles, size=num_health)
    selected_v_health = df_vehicles.iloc[v_health_indices].reset_index(drop=True)
    
    obs_times = [start_date + timedelta(seconds=int(s)) for s in np.random.randint(0, 365 * 24 * 3600, size=num_health)]
    age_months = np.random.uniform(1.0, 48.0, size=num_health)
    
    cycles = age_months * np.random.uniform(8.0, 20.0, size=num_health)
    capacity_fade = (cycles * 0.015) + (age_months * 0.08) + np.random.normal(0, 0.5, size=num_health)
    capacity_fade = np.clip(capacity_fade, 0.5, 25.0)
    soh = 100.0 - capacity_fade
    
    risk_labels = np.where(soh < 85.0, 'High', np.where(soh < 92.0, 'Moderate', 'Low'))
    
    df_health = pd.DataFrame({
        'health_record_id': health_ids,
        'vehicle_id': selected_v_health['vehicle_id'].values,
        'observation_time': obs_times,
        'battery_age_months': np.round(age_months, 1),
        'battery_soh_pct': np.round(soh, 2),
        'equivalent_full_cycles': np.round(cycles, 1),
        'cumulative_energy_throughput_kwh': np.round(cycles * selected_v_health['usable_battery_capacity_kwh'].values, 1),
        'capacity_fade_pct': np.round(capacity_fade, 2),
        'battery_degradation_risk': risk_labels
    })
    df_health.to_csv(os.path.join(output_dir, "ev_battery_health.csv"), index=False)

    # -------------------------------------------------------------
    # FILE 5: ev_charging_telemetry.csv (~1,500,000 rows)
    # -------------------------------------------------------------
    print("5/6 Generating ev_charging_telemetry.csv (Sampling first 50,000 sessions)...")
    telemetry_list = []
    sampled_sessions = df_sessions.head(50000)
    
    for idx, row in sampled_sessions.iterrows():
        dur_mins = row['session_duration_minutes']
        steps = int(np.clip(dur_mins // 5, 2, 30))  # 5-minute intervals
        
        elapsed = np.linspace(0, dur_mins, steps)
        soc_prog = np.linspace(row['soc_start_pct'], row['soc_end_pct'], steps)
        
        # Taper power above 80% SOC
        power_taper = np.where(soc_prog > 80.0, row['average_charging_power_kw'] * (1 - (soc_prog - 80.0)/40.0), row['average_charging_power_kw'])
        power_taper = np.clip(power_taper, 5.0, 350.0)
        
        for s in range(steps):
            telemetry_list.append({
                'session_id': row['session_id'],
                'vehicle_id': row['vehicle_id'],
                'station_id': row['station_id'],
                'elapsed_minutes': np.round(elapsed[s], 1),
                'soc_pct': np.round(soc_prog[s], 2),
                'charging_power_kw': np.round(power_taper[s], 2),
                'battery_temperature_c': np.round(row['ambient_temperature_c'] + 5.0 + (s * 0.4), 1)
            })
            
    df_telemetry = pd.DataFrame(telemetry_list)
    df_telemetry.to_csv(os.path.join(output_dir, "ev_charging_telemetry.csv"), index=False)

    # -------------------------------------------------------------
    # FILE 6: ev_station_hourly_demand.csv (250,000 rows)
    # -------------------------------------------------------------
    print("6/6 Generating ev_station_hourly_demand.csv...")
    num_demand = 250000
    station_demand_ids = np.random.choice(station_ids, size=num_demand)
    hourly_timestamps = [start_date + timedelta(hours=int(h)) for h in np.random.randint(0, 365 * 24, size=num_demand)]
    
    hours = [t.hour for t in hourly_timestamps]
    active_sessions = np.random.poisson(lam=np.where(np.isin(hours, [8, 9, 17, 18, 19]), 8, 3), size=num_demand)
    energy_demand = active_sessions * np.random.uniform(15.0, 45.0, size=num_demand)
    
    df_demand = pd.DataFrame({
        'station_id': station_demand_ids,
        'timestamp': hourly_timestamps,
        'hour': hours,
        'active_charging_sessions': active_sessions,
        'energy_demand_kwh': np.round(energy_demand, 2),
        'station_utilization_pct': np.round(np.clip(active_sessions / 12.0 * 100, 0, 100), 1),
        'peak_demand_flag': np.where(active_sessions >= 8, 1, 0)
    })
    df_demand.to_csv(os.path.join(output_dir, "ev_station_hourly_demand.csv"), index=False)

    print(f"\nAll 6 files successfully created in '{output_dir}' folder!")

if __name__ == "__main__":
    generate_full_ev_dataset()