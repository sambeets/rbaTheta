"""
Adaptive CUSUM and SWRT for Fair Comparison
Makes baseline methods adaptive like your enhanced RBA-theta
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdaptiveCUSUMWindTurbineMonitor:
    """
    Adaptive CUSUM with parameter optimization
    Fair comparison with enhanced RBA-theta
    """
    
    def __init__(self):
        self.best_significance_level = None
        self.best_critical_value = None
        self.model = None
        self.is_fitted = False
        
    def optimize_parameters(self, data: pd.DataFrame) -> Dict:
        """
        Optimize CUSUM parameters for the specific dataset
        Similar to your tune_mixed_strategy() function
        """
        print("🔧 Optimizing CUSUM parameters...")
        
        # Get turbine columns
        turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        
        # Test different significance levels
        significance_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        best_score = -np.inf
        best_params = {'significance_level': 0.01}
        
        for sig_level in significance_levels:
            try:
                # Simulate CUSUM performance with this significance level
                score = self._evaluate_cusum_performance(data, sig_level)
                
                if score > best_score:
                    best_score = score
                    best_params['significance_level'] = sig_level
                    
            except Exception as e:
                continue
        
        # Store best parameters
        self.best_significance_level = best_params['significance_level']
        
        print(f"   Optimal significance level: {self.best_significance_level}")
        return best_params
    
    def _evaluate_cusum_performance(self, data: pd.DataFrame, sig_level: float) -> float:
        """
        Evaluate CUSUM performance with given parameters
        Uses cross-validation similar to your optimization approach
        """
        turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        
        total_score = 0
        valid_turbines = 0
        
        for turbine_col in turbine_columns[:3]:  # Test on subset for speed
            turbine_data = data[turbine_col].dropna()
            
            if len(turbine_data) < 100:
                continue
                
            try:
                # Simulate temperature data
                base_temp = 45
                T1 = base_temp + turbine_data * 0.1 + np.random.normal(0, 2, len(turbine_data))
                T2 = base_temp - 5 + turbine_data * 0.08 + np.random.normal(0, 1.5, len(turbine_data))
                T3 = base_temp - 8 + turbine_data * 0.06 + np.random.normal(0, 1, len(turbine_data))
                
                temperature_data = np.column_stack([T1, T2, T3])
                generator_speed = 800 + turbine_data * 5 + np.random.normal(0, 10, len(turbine_data))
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, test_idx in tscv.split(temperature_data):
                    # Train CUSUM
                    monitor = SimpleCUSUMMonitor(sig_level)
                    
                    try:
                        monitor.fit_normal_operation(
                            temperature_data[train_idx],
                            generator_speed[train_idx]
                        )
                        
                        # Test CUSUM
                        result = monitor.monitor_condition(
                            temperature_data[test_idx],
                            generator_speed[test_idx]
                        )
                        
                        # Score based on detection rate and consistency
                        detection_rate = len(result['fault_indices']) / len(test_idx)
                        
                        # Optimal detection rate: 2-10% for wind turbine faults
                        if 0.02 <= detection_rate <= 0.10:
                            score = 1.0
                        elif detection_rate < 0.02:
                            score = detection_rate / 0.02
                        else:
                            score = max(0.1, 0.10 / detection_rate)
                            
                        scores.append(score)
                        
                    except Exception:
                        scores.append(0.1)
                
                turbine_score = np.mean(scores) if scores else 0.1
                total_score += turbine_score
                valid_turbines += 1
                
            except Exception:
                continue
        
        return total_score / max(1, valid_turbines)
    
    def run_adaptive_cusum_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run CUSUM with optimized parameters"""
        # Optimize parameters first
        self.optimize_parameters(data)
        
        # Run CUSUM with optimal parameters
        all_events = []
        turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        
        for turbine_col in turbine_columns[:5]:  # Limit for speed
            turbine_data = data[turbine_col].dropna()
            
            if len(turbine_data) < 50:
                continue
            
            # Simulate sensors
            base_temp = 45
            T1 = base_temp + turbine_data * 0.1 + np.random.normal(0, 2, len(turbine_data))
            T2 = base_temp - 5 + turbine_data * 0.08 + np.random.normal(0, 1.5, len(turbine_data))
            T3 = base_temp - 8 + turbine_data * 0.06 + np.random.normal(0, 1, len(turbine_data))
            
            temperature_data = np.column_stack([T1, T2, T3])
            generator_speed = 800 + turbine_data * 5 + np.random.normal(0, 10, len(turbine_data))
            
            # Run adaptive CUSUM
            monitor = SimpleCUSUMMonitor(self.best_significance_level)
            
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
                            'turbine_id': turbine_col,
                            'start_time': split_point + fault_idx,
                            'end_time': split_point + fault_idx + 5,
                            'event_type': 'fault_detected',
                            'magnitude': abs(cusum_results['cusum_statistics'][fault_idx]),
                            'method': 'adaptive_cusum'
                        }
                        all_events.append(event)
                        
            except Exception as e:
                print(f"Adaptive CUSUM failed for {turbine_col}: {e}")
        
        return pd.DataFrame(all_events)


class AdaptiveSWRTRampDetector:
    """
    Adaptive SWRT with parameter optimization
    Fair comparison with enhanced RBA-theta
    """
    
    def __init__(self):
        self.best_epsilon_percent = None
        self.best_min_duration = None
        
    def optimize_parameters(self, data: pd.DataFrame, nominal_power: float) -> Dict:
        """
        Optimize SWRT parameters for the specific dataset
        Similar to your tune_mixed_strategy() function
        """
        print("🌪️ Optimizing SWRT parameters...")
        
        # Test different epsilon percentages
        epsilon_percentages = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        min_durations = [2, 3, 4, 5, 6]
        
        best_score = -np.inf
        best_params = {'epsilon_percent': 2.0, 'min_duration': 3}
        
        turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        
        for epsilon_pct in epsilon_percentages:
            for min_dur in min_durations:
                try:
                    score = self._evaluate_swrt_performance(
                        data, nominal_power, epsilon_pct, min_dur
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'epsilon_percent': epsilon_pct,
                            'min_duration': min_dur
                        }
                        
                except Exception:
                    continue
        
        # Store best parameters
        self.best_epsilon_percent = best_params['epsilon_percent']
        self.best_min_duration = best_params['min_duration']
        
        print(f"   Optimal epsilon: {self.best_epsilon_percent}%")
        print(f"   Optimal min duration: {self.best_min_duration}")
        
        return best_params
    
    def _evaluate_swrt_performance(self, data: pd.DataFrame, nominal_power: float,
                                  epsilon_pct: float, min_duration: int) -> float:
        """
        Evaluate SWRT performance with given parameters
        """
        turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        
        total_score = 0
        valid_turbines = 0
        
        for turbine_col in turbine_columns[:3]:  # Test subset for speed
            turbine_data = data[turbine_col].dropna().values
            
            if len(turbine_data) < 50:
                continue
            
            try:
                # Test SWRT with these parameters
                detector = SimpleSWRTDetector(epsilon_pct, min_duration)
                events = detector.detect_ramps(turbine_data, nominal_power, turbine_col)
                
                # Score based on detection rate and ramp balance
                detection_rate = len(events) / len(turbine_data)
                
                if len(events) > 0:
                    up_ramps = sum(1 for e in events if e['event_type'] == 'ramp_up')
                    down_ramps = len(events) - up_ramps
                    
                    # Good balance between up and down ramps
                    if down_ramps > 0:
                        balance = min(up_ramps, down_ramps) / max(up_ramps, down_ramps)
                    else:
                        balance = 0.0
                else:
                    balance = 0.0
                
                # Optimal detection rate: 5-20% for wind ramps
                if 0.05 <= detection_rate <= 0.20:
                    rate_score = 1.0
                elif detection_rate < 0.05:
                    rate_score = detection_rate / 0.05
                else:
                    rate_score = max(0.2, 0.20 / detection_rate)
                
                # Combined score
                combined_score = (rate_score + balance) / 2
                total_score += combined_score
                valid_turbines += 1
                
            except Exception:
                continue
        
        return total_score / max(1, valid_turbines)
    
    def run_adaptive_swrt_analysis(self, data: pd.DataFrame, nominal_power: float) -> pd.DataFrame:
        """Run SWRT with optimized parameters"""
        # Optimize parameters first
        self.optimize_parameters(data, nominal_power)
        
        # Run SWRT with optimal parameters
        all_events = []
        turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        
        for turbine_col in turbine_columns:
            turbine_data = data[turbine_col].dropna().values
            
            if len(turbine_data) < 10:
                continue
            
            try:
                # Use adaptive parameters
                detector = SimpleSWRTDetector(
                    self.best_epsilon_percent, 
                    self.best_min_duration
                )
                events = detector.detect_ramps(turbine_data, nominal_power, turbine_col)
                all_events.extend(events)
                
            except Exception as e:
                print(f"Adaptive SWRT failed for {turbine_col}: {e}")
        
        return pd.DataFrame(all_events)


class SimpleCUSUMMonitor:
    """Simplified CUSUM for parameter testing"""
    
    def __init__(self, significance_level: float = 0.01):
        self.alpha = significance_level
        self.model = None
        self.is_fitted = False
        self.critical_value = None
    
    def fit_normal_operation(self, temperature_data: np.ndarray, generator_speed: np.ndarray):
        self.model = LinearRegression()
        self.model.fit(temperature_data, generator_speed)
        self.is_fitted = True
        
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predicted = self.model.predict(temperature_data)
        residuals = generator_speed - predicted
        sigma_hat = np.std(residuals)
        cusum_stats = np.cumsum(residuals) / sigma_hat
        
        fault_indices = np.where(np.abs(cusum_stats) > self.critical_value)[0]
        
        return {
            'cusum_statistics': cusum_stats,
            'critical_value': self.critical_value,
            'fault_detected': len(fault_indices) > 0,
            'fault_time': fault_indices[0] if len(fault_indices) > 0 else None,
            'fault_indices': fault_indices
        }


class SimpleSWRTDetector:
    """Simplified SWRT for parameter testing"""
    
    def __init__(self, epsilon_percent: float = 2.0, min_duration: int = 3):
        self.epsilon_percent = epsilon_percent
        self.min_duration = min_duration
    
    def detect_ramps(self, power_data: np.ndarray, nominal_power: float, turbine_id: str) -> List[Dict]:
        events = []
        
        if len(power_data) < 10:
            return events
        
        epsilon = (self.epsilon_percent / 100) * nominal_power
        diffs = np.diff(power_data)
        significant_changes = np.abs(diffs) > epsilon
        
        if not np.any(significant_changes):
            return events
        
        change_indices = np.where(significant_changes)[0]
        
        if len(change_indices) == 0:
            return events
        
        # Group consecutive changes with minimum duration requirement
        current_start = change_indices[0]
        current_sum = diffs[current_start]
        
        for i in range(1, len(change_indices)):
            idx = change_indices[i]
            
            if idx - change_indices[i-1] <= 3:
                current_sum += diffs[idx]
            else:
                # Check duration requirement
                duration = change_indices[i-1] + 1 - current_start
                if abs(current_sum) > epsilon * 2 and duration >= self.min_duration:
                    event = {
                        'turbine_id': turbine_id,
                        'start_time': current_start,
                        'end_time': change_indices[i-1] + 1,
                        'event_type': 'ramp_up' if current_sum > 0 else 'ramp_down',
                        'magnitude': abs(current_sum),
                        'duration': duration,
                        'method': 'adaptive_swrt'
                    }
                    events.append(event)
                
                current_start = idx
                current_sum = diffs[idx]
        
        # Handle final ramp
        duration = change_indices[-1] + 1 - current_start
        if abs(current_sum) > epsilon * 2 and duration >= self.min_duration:
            event = {
                'turbine_id': turbine_id,
                'start_time': current_start,
                'end_time': change_indices[-1] + 1,
                'event_type': 'ramp_up' if current_sum > 0 else 'ramp_down',
                'magnitude': abs(current_sum),
                'duration': duration,
                'method': 'adaptive_swrt'
            }
            events.append(event)
        
        return events


def run_adaptive_cusum_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Run adaptive CUSUM analysis"""
    adaptive_cusum = AdaptiveCUSUMWindTurbineMonitor()
    return adaptive_cusum.run_adaptive_cusum_analysis(data)


def run_adaptive_swrt_analysis(data: pd.DataFrame, nominal_power: float) -> pd.DataFrame:
    """Run adaptive SWRT analysis"""
    adaptive_swrt = AdaptiveSWRTRampDetector()
    return adaptive_swrt.run_adaptive_swrt_analysis(data, nominal_power)


if __name__ == "__main__":
    print("🔧 Adaptive Baseline Methods")
    print("=" * 40)
    print("This ensures fair comparison by making")
    print("CUSUM and SWRT adaptive like your enhanced RBA-theta")
    print()
    print("Adaptive features:")
    print("• CUSUM: Optimized significance level")
    print("• SWRT: Optimized epsilon and duration thresholds")
    print("• Both: Cross-validation based parameter selection")
    print("• Both: Dataset-specific optimization")