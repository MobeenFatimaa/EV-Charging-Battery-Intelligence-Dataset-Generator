import os
import sys
import pandas as pd
import numpy as np

class EVDatasetValidator:
    def __init__(self, data_dir="./ev_dataset"):
        self.data_dir = data_dir
        self.errors = []
        self.warnings = []
        self.files = {}
        
        self.required_files = [
            "ev_vehicles.csv",
            "ev_charging_stations.csv",
            "ev_charging_sessions.csv",
            "ev_battery_health.csv",
            "ev_charging_telemetry.csv",
            "ev_station_hourly_demand.csv"
        ]

    def log_error(self, message):
        self.errors.append(f"❌ [ERROR] {message}")

    def log_warning(self, message):
        self.warnings.append(f"⚠️ [WARNING] {message}")

    def load_datasets(self):
        print("📥 Loading dataset files...")
        for file_name in self.required_files:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                self.log_error(f"Missing required file: {file_name}")
            else:
                try:
                    self.files[file_name] = pd.read_csv(file_path)
                    print(f"  ✓ Loaded {file_name} ({len(self.files[file_name]):,} rows)")
                except Exception as e:
                    self.log_error(f"Failed to read {file_name}: {str(e)}")

    def validate_referential_integrity(self):
        print("\n🔗 Validating Foreign Key Relationships...")
        df_v = self.files.get("ev_vehicles.csv")
        df_s = self.files.get("ev_charging_stations.csv")
        df_ses = self.files.get("ev_charging_sessions.csv")
        df_h = self.files.get("ev_battery_health.csv")
        df_t = self.files.get("ev_charging_telemetry.csv")
        df_d = self.files.get("ev_station_hourly_demand.csv")

        # Check vehicle_id links
        if df_v is not None:
            valid_vehicles = set(df_v['vehicle_id'])
            
            if df_ses is not None:
                orphan_ses_v = set(df_ses['vehicle_id']) - valid_vehicles
                if orphan_ses_v:
                    self.log_error(f"Found {len(orphan_ses_v)} vehicle_ids in sessions that do not exist in ev_vehicles.csv")

            if df_h is not None:
                orphan_h_v = set(df_h['vehicle_id']) - valid_vehicles
                if orphan_h_v:
                    self.log_error(f"Found {len(orphan_h_v)} vehicle_ids in battery_health that do not exist in ev_vehicles.csv")

        # Check station_id links
        if df_s is not None:
            valid_stations = set(df_s['station_id'])
            
            if df_ses is not None:
                orphan_ses_s = set(df_ses['station_id']) - valid_stations
                if orphan_ses_s:
                    self.log_error(f"Found {len(orphan_ses_s)} station_ids in sessions that do not exist in ev_charging_stations.csv")

            if df_d is not None:
                orphan_d_s = set(df_d['station_id']) - valid_stations
                if orphan_d_s:
                    self.log_error(f"Found {len(orphan_d_s)} station_ids in hourly_demand that do not exist in ev_charging_stations.csv")

        # Check session_id links in telemetry
        if df_ses is not None and df_t is not None:
            valid_sessions = set(df_ses['session_id'])
            orphan_telemetry_s = set(df_t['session_id']) - valid_sessions
            if orphan_telemetry_s:
                self.log_error(f"Found {len(orphan_telemetry_s)} session_ids in telemetry that do not exist in ev_charging_sessions.csv")

    def validate_sessions_physics(self):
        print("\n⚡ Validating Physical & Mathematical Consistency in Sessions...")
        df_ses = self.files.get("ev_charging_sessions.csv")
        if df_ses is None:
            return

        # 1. Start/End SOC logic
        invalid_soc = df_ses[df_ses['soc_start_pct'] >= df_ses['soc_end_pct']]
        if len(invalid_soc) > 0:
            self.log_error(f"{len(invalid_soc)} sessions have soc_start >= soc_end")

        # 2. SOC change math consistency
        soc_calc_diff = np.abs((df_ses['soc_end_pct'] - df_ses['soc_start_pct']) - df_ses['soc_change_pct'])
        if (soc_calc_diff > 0.1).any():
            self.log_error("Mismatch detected between (soc_end - soc_start) and soc_change_pct field (>0.1% difference)")

        # 3. Energy Grid vs Battery Energy (Grid energy must exceed delivered energy)
        efficiency_mismatch = df_ses[df_ses['energy_drawn_from_grid_kwh'] < df_ses['energy_delivered_kwh']]
        if len(efficiency_mismatch) > 0:
            self.log_error(f"{len(efficiency_mismatch)} sessions report grid energy less than battery delivered energy (violates >100% efficiency law)")

        # 4. Duration vs Timestamps
        start_ts = pd.to_datetime(df_ses['session_start_time'])
        end_ts = pd.to_datetime(df_ses['session_end_time'])
        duration_calc = (end_ts - start_ts).dt.total_seconds() / 60.0
        dur_mismatch = np.abs(duration_calc - df_ses['session_duration_minutes'])
        if (dur_mismatch > 0.5).any():
            self.log_error("Session start/end timestamp delta does not match session_duration_minutes column")

    def validate_bounds_and_nulls(self):
        print("\n🔍 Checking Bounds, Non-Negativity, and Missing Values...")
        
        # Define expected value bounds: (min_val, max_val)
        bounds_map = {
            "ev_vehicles.csv": {
                'battery_capacity_kwh': (20, 200),
                'initial_battery_soh_pct': (70, 100),
                'rated_range_km': (50, 1000)
            },
            "ev_charging_sessions.csv": {
                'soc_start_pct': (0, 100),
                'soc_end_pct': (0, 100),
                'charging_efficiency_pct': (50, 100),
                'ambient_temperature_c': (-40, 60),
                'total_session_cost': (0, 300)
            },
            "ev_battery_health.csv": {
                'battery_soh_pct': (50, 100),
                'capacity_fade_pct': (0, 50)
            }
        }

        for fname, df in self.files.items():
            # Check nulls
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                self.log_warning(f"File {fname} contains {total_nulls} null values across columns: {null_counts[null_counts > 0].to_dict()}")
            
            # Check bounds
            if fname in bounds_map:
                for col, (min_b, max_b) in bounds_map[fname].items():
                    if col in df.columns:
                        out_of_bounds = df[(df[col] < min_b) | (df[col] > max_b)]
                        if len(out_of_bounds) > 0:
                            self.log_error(f"In {fname}, column '{col}' has {len(out_of_bounds)} values outside realistic bounds ({min_b}, {max_b})")

    def run(self):
        self.load_datasets()
        if not self.files:
            print("❌ Validation stopped: No dataset files were loaded.")
            return

        self.validate_referential_integrity()
        self.validate_sessions_physics()
        self.validate_bounds_and_nulls()

        print("\n" + "="*50)
        print(" 📊 VALIDATION REPORT SUMMARY")
        print("="*50)
        
        if not self.errors and not self.warnings:
            print("🎉 SUCCESS: All dataset files passed validation tests with 0 errors and 0 warnings!")
            sys.exit(0)
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  {w}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"  {e}")
            print("\n🚨 Validation failed. Please fix the above errors in your data generation script.")
            sys.exit(1)
        else:
            print("\n✅ Passed validation with minor warnings.")

if __name__ == "__main__":
    validator = EVDatasetValidator(data_dir="./ev_dataset")
    validator.run()