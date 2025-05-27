import sys
import traceback
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import datetime # Use this for timedelta consistently

from Simulator.DBAPI import utils as ut 
from Simulator.DBAPI.debug_utils import DebugLogger as dl
from Simulator.DBAPI.type_classes import AnomalySetting

# Import your specific anomaly injection method classes
from Simulator.AnomalyInjector.InjectionMethods.lowered import LoweredAnomaly
from Simulator.AnomalyInjector.InjectionMethods.offline import OfflineAnomaly
from Simulator.AnomalyInjector.InjectionMethods.custom import CustomAnomaly
from Simulator.AnomalyInjector.InjectionMethods.step import StepAnomaly
from Simulator.AnomalyInjector.InjectionMethods.spike import SpikeAnomaly

class TimeSeriesAnomalyInjector:
    def __init__(self, seed: int = 42, debug=False):
        self.rng = np.random.default_rng(seed)
        self.debug = debug
        dl.debug_print("TimeSeriesAnomalyInjector initialized.")
        sys.stdout.flush()

    def inject_anomaly(
        self, 
        data: pd.DataFrame,
        # anomaly_settings can be a single setting object when called from BatchImporter per setting
        anomaly_setting_obj: AnomalySetting,
        ) -> pd.DataFrame:
        
        dl.debug_print(f"TimeSeriesAnomalyInjector.inject_anomaly called. Data shape: {data.shape}")
        if 'timestamp' in data.columns:
            dl.debug_print(f"  Initial chunk timestamp dtype: {data['timestamp'].dtype}, tz: {data['timestamp'].dt.tz if pd.api.types.is_datetime64_any_dtype(data['timestamp']) else 'N/A'}")
        sys.stdout.flush()

        try:
            # Work on a copy to avoid modifying the DataFrame passed from BatchImporter directly
            # BatchImporter manages the iteration of settings and accumulation of changes.
            # This injector processes ONE setting at a time on a copy of the current chunk state.
            modified_data = data.copy(deep=True)

            # Ensure essential columns for anomaly flagging exist
            if 'injected_anomaly' not in modified_data.columns:
                modified_data['injected_anomaly'] = False
            # 'is_anomaly' is handled by BatchImporter after this method returns

            # Timestamp column is expected to be named 'timestamp' and be UTC datetime64[ns]
            timestamp_col = 'timestamp' 
            if timestamp_col not in modified_data.columns:
                dl.debug_print(f"  ERROR (Injector): Expected timestamp column '{timestamp_col}' not found. Columns: {modified_data.columns.tolist()}")
                sys.stdout.flush()
                return data # Return original on critical error
            if not pd.api.types.is_datetime64_any_dtype(modified_data[timestamp_col]) or \
               modified_data[timestamp_col].dt.tz is None or \
               str(modified_data[timestamp_col].dt.tz).upper() != 'UTC':
                dl.debug_print(f"  ERROR (Injector): Timestamp column '{timestamp_col}' is not UTC datetime64. Dtype: {modified_data[timestamp_col].dtype}, TZ: {modified_data[timestamp_col].dt.tz}. Cannot inject.")
                sys.stdout.flush()
                return data

            dl.debug_print(f"  Processing AnomalySetting: Type='{anomaly_setting_obj.anomaly_type}', Timestamp='{anomaly_setting_obj.timestamp}', Duration='{anomaly_setting_obj.duration}', Columns='{anomaly_setting_obj.columns}'")
            sys.stdout.flush()
            
            # anomaly_setting_obj.timestamp is already an absolute, UTC-aware pd.Timestamp
            if not isinstance(anomaly_setting_obj.timestamp, pd.Timestamp) or pd.isna(anomaly_setting_obj.timestamp) or anomaly_setting_obj.timestamp.tzinfo is None:
                dl.debug_print(f"    Skipping setting: Invalid or non-UTC absolute timestamp provided: {anomaly_setting_obj.timestamp}")
                sys.stdout.flush()
                return modified_data # Return current state of modified_data
            
            start_time_utc = anomaly_setting_obj.timestamp
            
            duration_timedelta: Optional[datetime.timedelta] = None
            try:
                # Use the corrected ut.parse_duration, which returns datetime.timedelta
                duration_timedelta = ut.parse_duration(str(anomaly_setting_obj.duration)) # Ensure duration is string
                if not isinstance(duration_timedelta, datetime.timedelta) or duration_timedelta.total_seconds() < 0:
                    # Allow zero duration for instantaneous anomalies if methods handle it.
                    # If duration must be strictly positive, change to <= 0
                    dl.debug_print(f"    Skipping setting: Invalid duration '{anomaly_setting_obj.duration}' (parsed as {duration_timedelta}, total_seconds: {duration_timedelta.total_seconds() if isinstance(duration_timedelta, datetime.timedelta) else 'N/A'}).")
                    sys.stdout.flush()
                    return modified_data
            except ValueError as e_dur_parse: 
                dl.debug_print(f"    Skipping setting: Error parsing duration '{anomaly_setting_obj.duration}': {e_dur_parse}")
                sys.stdout.flush()
                return modified_data
            except Exception as e_dur_generic: # Catch any other parsing error
                dl.debug_print(f"    Skipping setting: Generic error parsing duration '{anomaly_setting_obj.duration}': {e_dur_generic}")
                traceback.print_exc()
                sys.stdout.flush()
                return modified_data

            end_time_utc = start_time_utc + duration_timedelta
            
            dl.debug_print(f"    Anomaly span for current setting: [{start_time_utc}] to [{end_time_utc}] (UTC)")
            sys.stdout.flush()

            # Create span mask (data timestamps are already UTC)
            # Use < end_time_utc for typical interval [start, end)
            span_mask = (modified_data[timestamp_col] >= start_time_utc) & \
                        (modified_data[timestamp_col] < end_time_utc) 
            
            if not span_mask.any():
                dl.debug_print(f"    No data points in current chunk slice fall within anomaly span. No injection for this setting on this chunk.")
                sys.stdout.flush()
                return modified_data
                
            span_data_indices = modified_data[span_mask].index
            dl.debug_print(f"    Found {len(span_data_indices)} data points within anomaly span.")
            sys.stdout.flush()

            columns_to_affect = anomaly_setting_obj.columns
            percentage = float(anomaly_setting_obj.percentage) # Ensure float for calculation

            target_columns = []
            if columns_to_affect: 
                for col in columns_to_affect:
                    if col in modified_data.columns and pd.api.types.is_numeric_dtype(modified_data[col]):
                        target_columns.append(col)
                    else:
                        dl.debug_print(f"    Warning: Specified column '{col}' not found or not numeric. Skipping for this column.")
            else: 
                target_columns = [col for col in modified_data.select_dtypes(include=[np.number]).columns if col not in [timestamp_col, 'id', 'label', 'is_anomaly', 'injected_anomaly']]
            
            if not target_columns:
                dl.debug_print(f"    Warning: No valid target columns found for injection. Skipping setting application.")
                sys.stdout.flush()
                return modified_data
            
            dl.debug_print(f"    Target columns for injection: {target_columns}")
            sys.stdout.flush()

            for col_to_inject in target_columns:
                try:
                    num_points_in_span_for_col = len(span_data_indices) # Number of rows matching time span
                    num_anomalies_to_inject = min(num_points_in_span_for_col, max(0, int(num_points_in_span_for_col * (percentage / 100.0))))
                    
                    if num_anomalies_to_inject == 0:
                        dl.debug_print(f"    Skipping column '{col_to_inject}': Calculated num_anomalies_to_inject is 0.")
                        continue

                    anomaly_indices_for_column = self.rng.choice(span_data_indices, size=num_anomalies_to_inject, replace=False)
                    
                    dl.debug_print(f"      Injecting into column '{col_to_inject}' at {len(anomaly_indices_for_column)} randomly selected indices within span.")
                    dl.debug_print(f"        Sample indices: {anomaly_indices_for_column.tolist()[:5]}")
                    dl.debug_print(f"        Data BEFORE injection for '{col_to_inject}' at sample indices:\n{modified_data.loc[anomaly_indices_for_column, col_to_inject].head()}")
                    sys.stdout.flush()
                    
                    data_for_stats = modified_data.loc[anomaly_indices_for_column, col_to_inject]
                    data_range_val = data_for_stats.max() - data_for_stats.min() if not data_for_stats.empty and data_for_stats.notna().any() else 0.0
                    mean_val = data_for_stats.mean() if not data_for_stats.empty and data_for_stats.notna().any() else 0.0
                    
                    if anomaly_setting_obj.data_range is not None and len(anomaly_setting_obj.data_range) > 0 : data_range_val = float(anomaly_setting_obj.data_range[0])
                    if anomaly_setting_obj.mean is not None and len(anomaly_setting_obj.mean) > 0 : mean_val = float(anomaly_setting_obj.mean[0])

                    modified_series = self._apply_anomaly(
                        data_series_to_modify=modified_data.loc[anomaly_indices_for_column, col_to_inject].copy(), # Pass a copy of the series slice
                        data_range=data_range_val,
                        rng=self.rng, 
                        mean=mean_val,
                        settings_for_method=anomaly_setting_obj 
                    )
                    modified_data.loc[anomaly_indices_for_column, col_to_inject] = modified_series
                    
                    # Mark 'injected_anomaly' flag
                    modified_data.loc[anomaly_indices_for_column, "injected_anomaly"] = True
                    modified_data.loc[anomaly_indices_for_column, "label"] = 1 # Also update the labels
                    
                    dl.debug_print(f"        Data AFTER injection for '{col_to_inject}' at sample indices:\n{modified_data.loc[anomaly_indices_for_column, col_to_inject].head()}")
                    dl.debug_print(f"        'injected_anomaly' flags set for '{col_to_inject}' at sample indices:\n{modified_data.loc[anomaly_indices_for_column, 'injected_anomaly'].head()}")
                    sys.stdout.flush()

                except Exception as e_col:
                    dl.print_exception(f"    Error during anomaly application for column '{col_to_inject}': {e_col}")
                    sys.stdout.flush()
            
            return modified_data # Return the chunk modified by this one setting
        
        except Exception as e_main:
            dl.print_exception(f"CRITICAL Error in TimeSeriesAnomalyInjector.inject_anomaly: {e_main}")
            sys.stdout.flush()
            return data # Return original data on major failure

    def _apply_anomaly(self, 
                       data_series_to_modify: pd.Series, 
                       data_range: float, 
                       rng: np.random.Generator, 
                       mean: float, 
                       settings_for_method: AnomalySetting):
        try:
            anomaly_type = settings_for_method.anomaly_type
            magnitude = float(settings_for_method.magnitude) # Ensure magnitude is float

            dl.debug_print(f"      _apply_anomaly: Type='{anomaly_type}', Magnitude='{magnitude}', Data Series (head): \n{data_series_to_modify.head()}")
            sys.stdout.flush()

            if anomaly_type == 'lowered':
                injector = LoweredAnomaly()
                return injector.inject_anomaly(data_series_to_modify, rng, data_range, mean)
            elif anomaly_type == 'spike':
                injector = SpikeAnomaly()
                return injector.inject_anomaly(data_series_to_modify, rng, magnitude)
            elif anomaly_type == 'step':
                injector = StepAnomaly()
                return injector.inject_anomaly(data_series_to_modify, mean, magnitude)
            elif anomaly_type == 'offline': # This typically means data points are set to 0 or a fixed offline value
                injector = OfflineAnomaly()
                # OfflineAnomaly might set to 0 or a specific value, or take magnitude as that value
                return injector.inject_anomaly(data_series_to_modify, magnitude_is_offline_value=magnitude) 
            elif anomaly_type == 'custom':
                injector = CustomAnomaly()
                return injector.inject_anomaly(data_series_to_modify, magnitude) 
            else:
                dl.debug_print(f"      _apply_anomaly: Unknown anomaly type '{anomaly_type}'. Returning original data for series.")
                sys.stdout.flush()
                return data_series_to_modify
        except Exception as e:
            dl.print_exception(f"    Error in _apply_anomaly (type: {settings_for_method.anomaly_type}): {e}")
            sys.stdout.flush()
            return data_series_to_modify
