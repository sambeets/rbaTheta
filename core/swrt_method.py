"""
core/swrt_method.py
SWRT (Swinging Door/Wind Ramp) method
Copy this code into your core/ directory
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class SWRTRampDetector:
    """
    Simple Wind Ramp Threshold (SWRT) detector
    Based on Swinging Door Algorithm principles
    """
    
    def __init__(self, epsilon_percent: float = 2.0):
        """
        Initialize SWRT detector
        
        Args:
            epsilon_percent: Threshold as percentage of nominal power (default: 2%)
        """
        self.epsilon_percent = epsilon_percent
        self.ramp_events = []
        
    def detect_ramps(self, power_data: np.ndarray, nominal_power: float, turbine_id: str) -> List[Dict]:
        """
        Detect wind ramp events using SWRT method
        
        Args:
            power_data: Wind power time series
            nominal_power: Nominal power rating
            turbine_id: Turbine identifier
            
        Returns:
            List of detected ramp events
        """
        events = []
        
        if len(power_data) < 10:
            return events
        
        # Calculate threshold
        epsilon = (self.epsilon_percent / 100) * nominal_power
        
        # Find significant changes
        diffs = np.diff(power_data)
        significant_changes = np.abs(diffs) > epsilon
        
        if not np.any(significant_changes):
            return events
        
        # Group consecutive changes into ramps
        change_indices = np.where(significant_changes)[0]
        
        if len(change_indices) == 0:
            return events
        
        # Simple ramp grouping algorithm
        current_start = change_indices[0]
        current_sum = diffs[current_start]
        
        for i in range(1, len(change_indices)):
            idx = change_indices[i]
            
            # If changes are close together (within 3 time steps), group them
            if idx - change_indices[i-1] <= 3:
                current_sum += diffs[idx]
            else:
                # End current ramp if magnitude is significant
                if abs(current_sum) > epsilon * 2:  # Minimum ramp magnitude
                    ramp_event = self._create_ramp_event(
                        current_start, change_indices[i-1] + 1,
                        power_data, current_sum, turbine_id
                    )
                    events.append(ramp_event)
                
                # Start new ramp
                current_start = idx
                current_sum = diffs[idx]
        
        # Handle final ramp
        if abs(current_sum) > epsilon * 2:
            ramp_event = self._create_ramp_event(
                current_start, change_indices[-1] + 1,
                power_data, current_sum, turbine_id
            )
            events.append(ramp_event)
        
        return events
    
    def _create_ramp_event(self, start_idx: int, end_idx: int, power_data: np.ndarray, 
                          magnitude: float, turbine_id: str) -> Dict:
        """Create a ramp event dictionary"""
        duration = end_idx - start_idx
        start_power = power_data[start_idx]
        end_power = power_data[end_idx] if end_idx < len(power_data) else power_data[-1]
        
        return {
            'turbine_id': turbine_id,
            'start_time': start_idx,
            'end_time': end_idx,
            'event_type': 'ramp_up' if magnitude > 0 else 'ramp_down',
            'magnitude': abs(magnitude),
            'duration': duration,
            'start_power': start_power,
            'end_power': end_power,
            'ramp_rate': abs(magnitude) / duration if duration > 0 else 0,
            'method': 'swrt'
        }

def run_swrt_on_turbine_data(turbine_data: np.ndarray, nominal_power: float, turbine_id: str) -> List[Dict]:
    """
    Run SWRT analysis on single turbine data
    """
    detector = SWRTRampDetector(epsilon_percent=2.0)  # 2% threshold
    events = detector.detect_ramps(turbine_data, nominal_power, turbine_id)
    return events

def run_swrt_analysis(data: pd.DataFrame, nominal_power: float) -> pd.DataFrame:
    """
    Run SWRT analysis on all turbines in dataset
    Returns DataFrame of all detected ramp events
    """
    all_events = []
    
    # Get turbine columns
    turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
    
    for turbine_col in turbine_columns:
        turbine_data = data[turbine_col].dropna().values
        
        if len(turbine_data) < 10:
            continue
            
        turbine_events = run_swrt_on_turbine_data(turbine_data, nominal_power, turbine_col)
        all_events.extend(turbine_events)
    
    return pd.DataFrame(all_events)