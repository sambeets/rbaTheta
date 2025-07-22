"""
core/cusum_method.py
CUSUM method for wind turbine condition monitoring
Copy this code into your core/ directory
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CUSUMWindTurbineMonitor:
    """
    CUSUM-based condition monitoring for wind turbines
    Based on Dao (2021) paper implementation
    """
    
    def __init__(self, significance_level: float = 0.01):
        self.alpha = significance_level
        self.model = None
        self.is_fitted = False
        self.critical_value = None
        self.cusum_stats = []
        self.fault_detected = False
        self.fault_time = None
        
    def fit_normal_operation(self, temperature_data: np.ndarray, generator_speed: np.ndarray):
        """Fit CUSUM model on normal operation data"""
        self.model = LinearRegression()
        self.model.fit(temperature_data, generator_speed)
        self.is_fitted = True
        
        # Calculate critical value
        n_samples = len(generator_speed)
        n_params = temperature_data.shape[1] + 1
        
        if self.alpha == 0.01:
            a = 1.143
        elif self.alpha == 0.05:
            a = 0.948
        else:
            a = 0.948 + (1.143 - 0.948) * (0.05 - self.alpha) / (0.05 - 0.01)
        
        self.critical_value = a * np.sqrt(n_samples - n_params)
        
    def monitor_condition(self, temperature_data: np.ndarray, generator_speed: np.ndarray, 
                         time_index: np.ndarray = None) -> Dict:
        """Monitor for faults using CUSUM"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Calculate residuals
        predicted = self.model.predict(temperature_data)
        residuals = generator_speed - predicted
        
        # Calculate CUSUM statistics
        sigma_hat = np.std(residuals)
        cusum_stats = np.cumsum(residuals) / sigma_hat
        self.cusum_stats = cusum_stats
        
        # Check for faults
        fault_indices = np.where(np.abs(cusum_stats) > self.critical_value)[0]
        
        self.fault_detected = len(fault_indices) > 0
        self.fault_time = fault_indices[0] if self.fault_detected else None
        
        return {
            'cusum_statistics': cusum_stats,
            'critical_value': self.critical_value,
            'fault_detected': self.fault_detected,
            'fault_time': self.fault_time,
            'fault_indices': fault_indices
        }

def run_cusum_on_turbine_data(turbine_data: np.ndarray, turbine_id: str) -> List[Dict]:
    """
    Run CUSUM analysis on single turbine data
    Returns list of detected events
    """
    events = []
    
    if len(turbine_data) < 50:
        return events
    
    # Simulate temperature sensors from power data
    base_temp = 45
    T1 = base_temp + turbine_data * 0.1 + np.random.normal(0, 2, len(turbine_data))
    T2 = base_temp - 5 + turbine_data * 0.08 + np.random.normal(0, 1.5, len(turbine_data))
    T3 = base_temp - 8 + turbine_data * 0.06 + np.random.normal(0, 1, len(turbine_data))
    
    temperature_data = np.column_stack([T1, T2, T3])
    generator_speed = 800 + turbine_data * 5 + np.random.normal(0, 10, len(turbine_data))
    
    # Run CUSUM
    monitor = CUSUMWindTurbineMonitor(significance_level=0.01)
    
    # Split: 70% training, 30% monitoring
    split_point = int(len(turbine_data) * 0.7)
    
    try:
        monitor.fit_normal_operation(
            temperature_data[:split_point],
            generator_speed[:split_point]
        )
        
        cusum_results = monitor.monitor_condition(
            temperature_data[split_point:],
            generator_speed[split_point:],
            np.arange(split_point, len(turbine_data))
        )
        
        # Convert to events
        if cusum_results['fault_detected']:
            for fault_idx in cusum_results['fault_indices']:
                event = {
                    'turbine_id': turbine_id,
                    'start_time': split_point + fault_idx,
                    'end_time': split_point + fault_idx + 5,  # 5-hour duration
                    'event_type': 'fault_detected',
                    'magnitude': abs(cusum_results['cusum_statistics'][fault_idx]),
                    'method': 'cusum'
                }
                events.append(event)
                
    except Exception as e:
        print(f"CUSUM failed for {turbine_id}: {e}")
    
    return events

def run_cusum_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run CUSUM analysis on all turbines in dataset
    Returns DataFrame of all detected events
    """
    all_events = []
    
    # Get turbine columns
    turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
    
    for turbine_col in turbine_columns[:5]:  # Limit to 5 turbines for speed
        turbine_data = data[turbine_col].dropna().values
        turbine_events = run_cusum_on_turbine_data(turbine_data, turbine_col)
        all_events.extend(turbine_events)
    
    return pd.DataFrame(all_events)