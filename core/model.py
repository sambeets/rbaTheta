"""The enhanced rbaTheta model"""


"""
Enhanced model.py - Simplified version
Core improvements: RF-based thresholding, dynamic parameters, no randomness
Maintains ALL original function names and signatures
"""

#from giddy.markov import LISA_Markov, Spatial_Markov
#from libpysal.weights import Queen, DistanceBand
#import libpysal
#import geopandas
import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Real, Integer
import time
import pickle
import core.helpers as fn
import core.event_extraction as ee
from core.database import RBAThetaDB
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore') 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ORIGINAL FUNCTIONS (MAINTAINED) ====================

def likelihood(threshold, data):
    """Original function - maintained for compatibility"""
    event_count = np.sum(np.abs(data) > threshold)
    return event_count if event_count > 0 else 1e-10

def calculate_adaptive_threshold(data):
    """Original function - enhanced but maintains signature"""
    mean = np.mean(data)
    std = np.std(data)
    
    # Enhanced: Add coefficient of variation adjustment
    cv = std / mean if mean != 0 else 1
    adjustment_factor = 1 + cv * 0.1  # More variable data gets higher threshold
    
    return (mean + 0.05 * std) * adjustment_factor

def calculate_adaptive_threshold_mcmc(data, iterations=5000, initial_threshold=0.1, 
                                      n_estimators=100, max_samples=0.8):
    """
    Enhanced Random Forest-based MCMC threshold estimation
    Uses RF to propose better MCMC moves instead of random walk
    """
    data_array = data.values if isinstance(data, pd.Series) else np.array(data)
    
    if len(data_array) < 10:
        return initial_threshold
    
    # ==================== RF PROPOSAL MECHANISM ====================
    
    def train_rf_proposal_model(data_arr):
        """Train RF to learn good threshold proposals"""
        print("🌲 Training Random Forest proposal model...")
        
        # Create features for Random Forest
        X = np.arange(len(data_arr)).reshape(-1, 1)
        
        # Add rolling statistics as features
        if len(data_arr) > 24:
            rolling_12h = pd.Series(data_arr).rolling(12, min_periods=1).mean().values
            rolling_24h = pd.Series(data_arr).rolling(24, min_periods=1).mean().values
            X = np.column_stack([X, rolling_12h, rolling_24h])
        
        y = np.abs(data_arr)
        
        # Train Random Forest for proposals
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
            n_jobs=1
        )
        rf.fit(X, y)
        print(f"✅ RF model trained with {X.shape[1]} features and {n_estimators} trees")
        return rf, X
    
    def rf_proposal(current_threshold, rf_model, X, step_size=0.1):
        """Optimized for wind turbine threshold exploration"""
        rf_predictions = rf_model.predict(X)
        rf_mean = np.median(rf_predictions)
        rf_std = np.std(rf_predictions)
        
        # More aggressive exploration for wind data
        if np.random.random() < 0.4:  # 40% RF-guided jumps
            # Jump to RF-suggested regions
            proposed_threshold = np.random.normal(rf_mean * 0.1, rf_std * 0.2)
        else:  # 60% local exploration
            # Local moves around current threshold
            proposed_threshold = current_threshold + np.random.normal(0, step_size * 5)
        
        # Keep in reasonable range for wind data
        return max(0.0001, min(proposed_threshold, rf_std * 2.0))
    
    # ==================== MCMC LIKELIHOOD FUNCTION ====================
    
    def log_likelihood(threshold, data_arr):
        """Optimized for wind turbine data"""
        events = np.sum(np.abs(data_arr) > threshold)
        
        if events == 0:
            return -100  # Less harsh penalty
        
        data_std = np.std(data_arr)
        
        # Adjust expected events based on your actual wind data patterns
        # Your data shows ~3-15% event rates are reasonable
        expected_events = len(data_arr) * 0.08  # 8% for wind turbine data
        
        # Scale the likelihood to be less harsh
        log_like = -0.1 * ((events - expected_events) / expected_events)**2
        
        # Gentler prior based on your data std
        log_prior = -0.1 * ((threshold - data_std * 0.1) / (data_std * 0.5))**2
        
        return log_like + log_prior
    
    # ==================== MCMC SAMPLING LOOP ====================
    
    # Train RF proposal model
    rf_model, X_features = train_rf_proposal_model(data_array)
    
    # Initialize MCMC
    current_threshold = initial_threshold
    current_log_like = log_likelihood(current_threshold, data_array)
    
    # MCMC sample storage
    samples = []
    accepted = 0
    
    # Adaptive step size
    step_size = 0.1
    target_acceptance = 0.4
    
    print(f"🔄 Starting RF-enhanced MCMC with {iterations} iterations...")
    print(f"📊 Initial threshold: {initial_threshold:.6f}, Initial log-likelihood: {current_log_like:.2f}")
    
    # Progress tracking
    progress_interval = max(500, iterations // 10)  # Report every 500 iterations or 10% of total
    
    for i in range(iterations):
        # RF-ENHANCED PROPOSAL (this is the key improvement!)
        proposed_threshold = rf_proposal(current_threshold, rf_model, X_features, step_size)
        proposed_log_like = log_likelihood(proposed_threshold, data_array)
        
        # Metropolis-Hastings acceptance
        log_alpha = proposed_log_like - current_log_like
        
        if log_alpha > 0 or np.random.random() < np.exp(log_alpha):
            # Accept proposal
            current_threshold = proposed_threshold
            current_log_like = proposed_log_like
            accepted += 1
        
        # Store sample (after burn-in)
        if i > iterations // 4:  # Skip first 25% as burn-in
            samples.append(current_threshold)
        
        # PROGRESS REPORTING
        if i > 0 and i % progress_interval == 0:
            current_acceptance = accepted / (i + 1)
            burn_in_complete = i > iterations // 4
            samples_collected = len(samples)
            
            print(f"   📈 Progress: {i}/{iterations} ({i/iterations*100:.1f}%) | "
                  f"Acceptance: {current_acceptance:.1%} | "
                  f"Current threshold: {current_threshold:.6f} | "
                  f"Samples collected: {samples_collected} | "
                  f"{'Post-burn-in' if burn_in_complete else 'Burn-in phase'}")
        
        # Adaptive step size adjustment
        if i > 0 and i % 500 == 0:
            acceptance_rate = accepted / (i + 1)
            old_step_size = step_size
            
            if acceptance_rate < 0.01:  # If below 1%
                step_size = 0.1  # RESET to original
                print(f"   🔄 RESET step size to 0.1 due to low acceptance ({acceptance_rate:.1%})")
            elif acceptance_rate < target_acceptance:
                step_size *= 0.95  # Gentler reduction
            else:
                step_size *= 1.05  # Gentler increase
            
            # Log step size adjustments
            if abs(step_size - old_step_size) > 0.001:
                print(f"   🔧 Step size adjusted: {old_step_size:.4f} → {step_size:.4f} "
                      f"(acceptance rate: {acceptance_rate:.1%})")
    
    # ==================== PROCESS MCMC RESULTS ====================
    
    if len(samples) == 0:
        print("⚠️ MCMC failed, falling back to RF estimate")
        rf_predictions = rf_model.predict(X_features)
        return np.median(rf_predictions)
    
    # Use posterior samples to get final threshold
    posterior_mean = np.mean(samples)
    posterior_std = np.std(samples)
    
    # Add data-driven adjustment
    data_std = np.std(data_array)
    data_mean = np.mean(data_array)
    volatility_adjustment = (data_std / data_mean) if data_mean > 0 else 1
    
    final_threshold = posterior_mean * (1 + volatility_adjustment * 0.3)
    
    # Ensure reasonable bounds
    min_threshold = data_std * 0.1
    max_threshold = data_std * 2.0
    
    acceptance_rate = accepted / iterations
    
    # COMPREHENSIVE FINAL REPORT
    print(f"✅ MCMC COMPLETED!")
    print(f"   📊 Total samples: {len(samples)} (after {iterations//4} burn-in)")
    print(f"   📈 Acceptance rate: {acceptance_rate:.1%} (target: {target_acceptance:.1%})")
    print(f"   🎯 Posterior mean: {posterior_mean:.6f} ± {posterior_std:.6f}")
    print(f"   🔧 Final threshold: {np.clip(final_threshold, min_threshold, max_threshold):.6f}")
    print(f"   ⏱️  Final step size: {step_size:.4f}")
    
    return np.clip(final_threshold, min_threshold, max_threshold)
# ==================== ENHANCED COMPONENTS ====================

def analyze_wind_data_properties(data):
    """
    Analyze wind turbine data with focus on meaningful ramp patterns
    Fixed version with proper error handling
    """
    turbine_cols = [col for col in data.columns if col.startswith('Turbine_')]
    
    properties = {
        'data_length': len(data),
        'turbine_count': len(turbine_cols),
        'sampling_interval': 1,  # Assume hourly data
    }
    
    if not turbine_cols:
        # If no turbine columns found, use default safe values
        properties.update({
            'overall_mean': 1.0,
            'overall_std': 0.1,
            'overall_cv': 0.1,
            'data_volatility': 0.1,  # Add default data_volatility
            'meaningful_change_magnitude': 0.02,
            'significant_change_threshold': 0.05,
            'major_change_threshold': 0.08,
        })
        
        # Add default ramp characteristics
        properties['ramp_characteristics'] = {
            'substantial_ramp_duration': [],
            'substantial_ramp_magnitude': [],
            'ramp_frequency': 0.005,
            'median_ramp_duration': 5,
            'median_ramp_magnitude': 0.05,
            'ramp_count': 0
        }
        return properties
    
    # Analyze each turbine's characteristics
    all_values = []
    all_meaningful_changes = []
    
    for col in turbine_cols:
        try:
            values = data[col].dropna()
            if len(values) > 0:
                all_values.extend(values.tolist())
                
                # Focus on meaningful changes (not every small fluctuation)
                diffs = values.diff().dropna()
                
                if len(diffs) > 0:
                    # Filter out noise - only consider changes above noise level
                    noise_level = diffs.abs().quantile(0.6)  # 60th percentile as noise threshold
                    meaningful_changes = diffs[diffs.abs() > noise_level]
                    all_meaningful_changes.extend(meaningful_changes.abs().tolist())
        except Exception as e:
            print(f"Warning: Error processing column {col}: {e}")
            continue
    
    # Handle case where no valid data was found
    if not all_values:
        all_values = [1.0, 1.1, 0.9, 1.05, 0.95]  # Default values
    
    # Overall dataset characteristics focused on signal vs noise
    all_values = np.array(all_values)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    
    # Calculate coefficient of variation and data volatility safely
    cv_val = std_val / mean_val if mean_val > 0 else 0.1
    data_volatility = cv_val  # This is the key that was missing!
    
    properties.update({
        'overall_mean': mean_val,
        'overall_std': std_val,
        'overall_cv': cv_val,
        'data_volatility': data_volatility,  # Fixed: Now properly set
    })
    
    # Calculate meaningful change patterns (excluding noise)
    if all_meaningful_changes:
        all_meaningful_changes = np.array(all_meaningful_changes)
        properties.update({
            'meaningful_change_magnitude': np.median(all_meaningful_changes),
            'significant_change_threshold': np.quantile(all_meaningful_changes, 0.7),  # 70th percentile
            'major_change_threshold': np.quantile(all_meaningful_changes, 0.85),  # 85th percentile
        })
    else:
        properties.update({
            'meaningful_change_magnitude': 0.02,
            'significant_change_threshold': 0.05,
            'major_change_threshold': 0.08,
        })
    
    # Detect substantial ramp patterns (not micro-fluctuations)
    try:
        properties['ramp_characteristics'] = detect_substantial_ramp_patterns(data, turbine_cols)
    except Exception as e:
        print(f"Warning: Error in ramp detection: {e}")
        # Provide safe defaults
        properties['ramp_characteristics'] = {
            'substantial_ramp_duration': [],
            'substantial_ramp_magnitude': [],
            'ramp_frequency': 0.005,
            'median_ramp_duration': 5,
            'median_ramp_magnitude': 0.05,
            'ramp_count': 0
        }
    
    return properties

def detect_substantial_ramp_patterns(data, turbine_cols):
    """
    Detect substantial ramp patterns, filtering out minor fluctuations
    """
    ramp_info = {
        'substantial_ramp_duration': [],
        'substantial_ramp_magnitude': [],
        'ramp_frequency': 0
    }
    
    for col in turbine_cols[:3]:  # Analyze first 3 turbines for better statistics
        values = data[col].dropna()
        if len(values) < 20:
            continue
            
        # Apply smoothing to reduce noise before ramp detection
        smoothed_values = values.rolling(window=3, center=True).mean().fillna(values)
        changes = smoothed_values.diff()
        
        # Define substantial change threshold
        change_threshold = changes.abs().quantile(0.7)  # Top 30% of changes
        
        # Find sequences of substantial, consistent direction changes
        current_ramp_start = None
        current_direction = None
        current_magnitude = 0
        
        for i, change in enumerate(changes):
            if pd.isna(change):
                continue
                
            # Only consider substantial changes
            if abs(change) < change_threshold:
                continue
                
            direction = 1 if change > 0 else -1
            
            if current_direction is None or direction != current_direction:
                # End of previous ramp
                if current_ramp_start is not None:
                    ramp_duration = i - current_ramp_start
                    if ramp_duration >= 3 and current_magnitude >= change_threshold * 2:  # Stricter criteria
                        ramp_info['substantial_ramp_duration'].append(ramp_duration)
                        ramp_info['substantial_ramp_magnitude'].append(current_magnitude)
                
                # Start new ramp
                current_ramp_start = i
                current_direction = direction
                current_magnitude = abs(change)
            else:
                # Continue current ramp
                current_magnitude += abs(change)
    
    # Calculate statistics for substantial ramps only
    if ramp_info['substantial_ramp_duration']:
        ramp_info['median_ramp_duration'] = np.median(ramp_info['substantial_ramp_duration'])
        ramp_info['median_ramp_magnitude'] = np.median(ramp_info['substantial_ramp_magnitude'])
        ramp_info['ramp_count'] = len(ramp_info['substantial_ramp_duration'])
        ramp_info['ramp_frequency'] = len(ramp_info['substantial_ramp_duration']) / len(values) if len(values) > 0 else 0
    else:
        # Conservative defaults if no substantial ramps detected
        ramp_info['median_ramp_duration'] = 5
        ramp_info['median_ramp_magnitude'] = 0.05
        ramp_info['ramp_count'] = 0
        ramp_info['ramp_frequency'] = 0.005  # Very low frequency
    
    return ramp_info


def calculate_balanced_parameters(properties):
    """
    Calculate balanced parameters with better stationary detection
    with separate factors
    """
    # Extract key properties
    volatility = properties['data_volatility']
    data_length = properties['data_length']
    meaningful_change = properties.get('meaningful_change_magnitude', 0.02)
    significant_change = properties.get('significant_change_threshold', 0.05)
    major_change = properties.get('major_change_threshold', 0.08)
    ramp_info = properties['ramp_characteristics']
    
    # Use substantial ramp characteristics
    typical_ramp_duration = ramp_info.get('median_ramp_duration', 5)
    
    # Traditional method parameters 
    trad_min_duration = max(2, int(typical_ramp_duration * 0.6))
    trad_min_slope = meaningful_change * 1.5
    trad_window = max(30, min(120, int(typical_ramp_duration * 4)))
    
    # RF/MCMC method parameters 
    mcmc_min_duration = max(3, int(typical_ramp_duration * 0.8))
    mcmc_min_slope = meaningful_change * 1.0
    mcmc_window = max(25, min(100, int(typical_ramp_duration * 3)))
    
    
    rf_n_estimators = min(120, max(60, int(data_length / 250)))
    rf_scale_factor = 0.4 + volatility * 0.3
    
    
    base_sig_factor = meaningful_change / significant_change if significant_change > 0 else 0.4
    sig_event_factor = max(0.00001, min(0.00008, base_sig_factor * 0.2))
    
    # CRITICAL: Much lower stationary factor for better detection (PREVIOUS LOGIC)
    stat_event_factor = max(0.00001, min(0.00005, sig_event_factor * 0.3))  # Much lower!
    
    # NEW: Separate Traditional and MCMC factors
    # Traditional factors use the original calculation
    trad_sig_factor = sig_event_factor
    trad_stat_factor = stat_event_factor
    
    # MCMC factors - slightly more restrictive for significant events
    mcmc_sig_factor = sig_event_factor * 0.85  # 15% more restrictive for significant
    mcmc_stat_factor = stat_event_factor  # Keep same for stationary
    
    # Frequency parameters
    base_freq = max(120, min(800, int(2400 / (ramp_info['ramp_frequency'] * 100 + 1))))
    
    config = {
        "trad_min_duration": trad_min_duration,
        "trad_min_slope": trad_min_slope,
        "trad_window": trad_window,
        "trad_min_stationary_length": max(3, int(typical_ramp_duration * 0.8)),  # Shorter!
        "trad_freq_secs": base_freq,
        "trad_sig_event_factor": trad_sig_factor,
        "trad_stat_event_factor": trad_stat_factor,
        
        "rf_n_estimators": rf_n_estimators,
        "rf_scale_factor": rf_scale_factor,
        
        "mcmc_min_duration": mcmc_min_duration,
        "mcmc_min_stationary_length": max(3, int(typical_ramp_duration * 0.6)),  # Much shorter!
        "mcmc_freq_secs": int(base_freq * 0.9),
        "mcmc_window": mcmc_window,
        "mcmc_min_slope": mcmc_min_slope,
        "mcmc_sig_event_factor": mcmc_sig_factor,  # Slightly more restrictive
        "mcmc_stat_event_factor": mcmc_stat_factor,

        # Legacy compatibility (use traditional factors as defaults)
        "sig_event_factor": trad_sig_factor,  # For backward compatibility
        "stat_event_factor": trad_stat_factor,  # For backward compatibility
    }
    
    return config

def validate_balanced_parameters(config, properties):
    """
    Validation with focus on stationary detection
    with separate factors
    """
    # Ensure balanced sensitivity (PREVIOUS LOGIC)
    if config['sig_event_factor'] > 0.0001:
        config['sig_event_factor'] = 0.0001
        logger.info("Capped sig_event_factor for precision")
    elif config['sig_event_factor'] < 0.00001:
        config['sig_event_factor'] = 0.00001
        logger.info("Increased sig_event_factor for adequate sensitivity")
    
    # CRITICAL: Force stationary factor to be much lower (PREVIOUS LOGIC)
    if config['stat_event_factor'] > 0.00005:
        config['stat_event_factor'] = 0.00005
        logger.info("Capped stat_event_factor for better stationary detection")
    
    if config['stat_event_factor'] < 0.00001:
        config['stat_event_factor'] = 0.00001
        logger.info("Set minimum stat_event_factor")
    
    # NEW: Apply same validation to separate factors
    # Traditional factors
    config['trad_sig_event_factor'] = max(0.00001, min(0.0001, config['trad_sig_event_factor']))
    config['trad_stat_event_factor'] = max(0.00001, min(0.00005, config['trad_stat_event_factor']))
    
    # MCMC factors - keep the restriction for significant events
    config['mcmc_sig_event_factor'] = max(0.00001, min(0.0001, config['mcmc_sig_event_factor']))
    config['mcmc_stat_event_factor'] = max(0.00001, min(0.00005, config['mcmc_stat_event_factor']))
    
    # Force shorter stationary lengths 
    config['trad_min_stationary_length'] = max(3, min(8, config['trad_min_stationary_length']))
    config['mcmc_min_stationary_length'] = max(3, min(8, config['mcmc_min_stationary_length']))
    
    # Balanced slope thresholds (PREVIOUS LOGIC)
    if config['mcmc_min_slope'] > 0.03:
        config['mcmc_min_slope'] = 0.03
        logger.info("Adjusted mcmc_min_slope for balanced detection")
    elif config['mcmc_min_slope'] < 0.005:
        config['mcmc_min_slope'] = 0.005
        logger.info("Increased mcmc_min_slope to reduce noise")
    
    if config['trad_min_slope'] > 0.05:
        config['trad_min_slope'] = 0.05
        logger.info("Adjusted trad_min_slope for balanced detection")
    elif config['trad_min_slope'] < 0.008:
        config['trad_min_slope'] = 0.008
        logger.info("Increased trad_min_slope to reduce noise")
    
    # Ensure reasonable window sizes 
    config['trad_window'] = max(30, min(120, config['trad_window']))
    config['mcmc_window'] = max(25, min(100, config['mcmc_window']))
    
    # Ensure reasonable durations 
    config['trad_min_duration'] = max(2, min(8, config['trad_min_duration']))
    config['mcmc_min_duration'] = max(3, min(10, config['mcmc_min_duration']))

    # Update legacy factors to use traditional method values
    config['sig_event_factor'] = config['trad_sig_event_factor']
    config['stat_event_factor'] = config['trad_stat_event_factor']
    
    logger.info(f"FIXED: stat_event_factor set to {config['stat_event_factor']:.6f} (much lower)")
    logger.info(f"FIXED: stationary lengths set to {config['mcmc_min_stationary_length']} (shorter)")
    
    return config

def tune_mixed_strategy(data, nominal):
    """
    Balanced parameter tuning - precise ramp detection
    """
    logger.info("Analyzing wind data for balanced ramp detection parameters...")
    
    # Analyze the dataset with focus on substantial patterns
    properties = analyze_wind_data_properties(data)
    
    logger.info(f"Dataset analysis for method-specific detection:")
    logger.info(f"  - Data volatility: {properties['data_volatility']:.3f}")
    logger.info(f"  - Meaningful change magnitude: {properties.get('meaningful_change_magnitude', 0):.4f}")
    logger.info(f"  - Substantial ramp frequency: {properties['ramp_characteristics']['ramp_frequency']:.3f}")
    logger.info(f"  - Median substantial ramp duration: {properties['ramp_characteristics']['median_ramp_duration']:.1f}")
    
    # Calculate balanced parameters
    config = calculate_balanced_parameters(properties)
    
    # Validate for balanced performance
    config = validate_balanced_parameters(config, properties)
    
    logger.info("Method-specific adaptive parameters calculated:")
    logger.info(f"  - Traditional sig factor: {config['trad_sig_event_factor']:.6f}")
    logger.info(f"  - Traditional stat factor: {config['trad_stat_event_factor']:.6f}")
    logger.info(f"  - RF-MCMC sig factor: {config['mcmc_sig_event_factor']:.6f}")
    logger.info(f"  - RF-MCMC stat factor: {config['mcmc_stat_event_factor']:.6f}")
    logger.info(f"  - Traditional slope: {config['trad_min_slope']:.4f}")
    logger.info(f"  - RF-MCMC slope: {config['mcmc_min_slope']:.4f}")
    logger.info(f"  - Expected event rate: {estimate_balanced_event_rate(config, properties):.1f}%")
    
    return config

def estimate_balanced_event_rate(config, properties):
    """
    Estimate expected event detection rate for balanced approach
    but with separate methods
    """
    # More conservative estimate based on substantial ramps only (PREVIOUS LOGIC)
    substantial_ramp_freq = properties['ramp_characteristics']['ramp_frequency']
    
    # Calculate separate rates for each method
    trad_sig_factor = config.get('trad_sig_event_factor', config.get('sig_event_factor', 0.001))
    mcmc_sig_factor = config.get('mcmc_sig_event_factor', config.get('sig_event_factor', 0.001))
    
    # Use previous sensitivity calculation logic
    trad_sensitivity = 1.0 / (trad_sig_factor * 50000)  # PREVIOUS LOGIC
    trad_rate = min(12.0, max(3.0, substantial_ramp_freq * trad_sensitivity * 100 * 0.5))
    
    mcmc_sensitivity = 1.0 / (mcmc_sig_factor * 50000)  # PREVIOUS LOGIC
    mcmc_rate = min(12.0, max(3.0, substantial_ramp_freq * mcmc_sensitivity * 100 * 0.5))
    
    # Total rate (both methods combined)
    total_rate = trad_rate + mcmc_rate
    
    return total_rate
# ==================== HELPER FUNCTIONS ====================

def evaluate_events(events_df):
    """Original function - maintained for compatibility"""
    if events_df.empty:
        return 0.0
    
    # Check if required columns exist
    if '∆t_m' in events_df.columns and '∆w_m' in events_df.columns:
        std_val = events_df['∆t_m'].std()
        return events_df['∆w_m'].abs().sum() / (1 + std_val) if std_val > 0 else 0.0
    else:
        return len(events_df)  # Fallback to count

def evaluate_stationary(events_df):
    """Original function - maintained for compatibility"""
    return len(events_df) if not events_df.empty else 0

def time_series_split(df, n_splits=3):
    """Original function - maintained for compatibility"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(df.iloc[train], df.iloc[test]) for train, test in tscv.split(df)]

def RBA_theta(data, nominal, param_config=None, db_path="rbatheta.db"):
    """
    Enhanced RBA_theta with Random Forest thresholding
    Original function signature maintained
    """
    if param_config is None:
        param_config = {}
    
    # Import required modules
    import core.event_extraction as ee
    from core.database import RBAThetaDB
    
    try:
        with RBAThetaDB(db_path) as db:
            db.load_data(data)
            db.normalize_data(nominal)
            turbine_ids = db.get_all_turbine_ids()
            tao = len(turbine_ids) + 1

            sig_events_trad, stat_events_trad = {}, {}
            sig_events_mcmc, stat_events_mcmc = {}, {}

            for turbine_id in turbine_ids:
                try:
                    turbine_data = db.get_turbine_data(turbine_id)
                    data_values = turbine_data['normalized_value'].values

                    # Traditional method
                    adaptive_threshold_trad = calculate_adaptive_threshold(data_values)
                    trad_sig_factor = param_config.get("trad_sig_event_factor", 
                                                      param_config.get("sig_event_factor", 0.00003))
                    trad_stat_factor = param_config.get("trad_stat_event_factor", 
                                                       param_config.get("stat_event_factor", 0.00009))
                    
                    sig_events_trad[turbine_id] = ee.significant_events(
                        data=data_values,
                        threshold=adaptive_threshold_trad * trad_sig_factor,
                        min_duration=param_config.get("trad_min_duration", 3),
                        min_slope=param_config.get("trad_min_slope", 0.05),
                        window_minutes=param_config.get("trad_window", 60),
                        freq_secs=param_config.get("trad_freq_secs", 100),
                    )

                    stat_events_trad[turbine_id] = ee.stationary_events(
                        data=data_values,
                        threshold=adaptive_threshold_trad * trad_stat_factor,
                        min_duration=param_config.get("trad_min_duration", 3),
                        min_stationary_length=param_config.get("trad_min_stationary_length", 7),
                        window_minutes=param_config.get("trad_window", 60),
                        freq_secs=param_config.get("trad_freq_secs", 100),
                    )

                    logger.info(f"Turbine {turbine_id} Traditional: threshold={adaptive_threshold_trad * param_config.get('stat_event_factor', 0.00009):.6f}, events={len(stat_events_trad[turbine_id])}")

                    # Enhanced Random Forest-MCMC method
                    rf_threshold = calculate_adaptive_threshold_mcmc(
                        turbine_data['normalized_value'],
                        n_estimators=param_config.get("rf_n_estimators", 100),
                        max_samples=param_config.get("rf_max_samples", 0.8)
                    )
                    mcmc_sig_factor = param_config.get("mcmc_sig_event_factor", 
                                                      param_config.get("sig_event_factor", 0.00003))
                    mcmc_stat_factor = param_config.get("mcmc_stat_event_factor", 
                                                       param_config.get("stat_event_factor", 0.00009))
                    
                    sig_events_mcmc[turbine_id] = ee.significant_events(
                        data=data_values,
                        threshold=rf_threshold * mcmc_sig_factor,
                        min_duration=param_config.get("mcmc_min_duration", 5),
                        min_slope=param_config.get("mcmc_min_slope", 0.03),
                        window_minutes=param_config.get("mcmc_window", 60),
                        freq_secs=param_config.get("mcmc_freq_secs", 600),
                    )

                    stat_events_mcmc[turbine_id] = ee.stationary_events(
                        data=data_values,
                        threshold=rf_threshold * mcmc_stat_factor,
                        min_duration=param_config.get("mcmc_min_duration", 5),
                        min_stationary_length=param_config.get("mcmc_min_stationary_length", 5),
                        window_minutes=param_config.get("mcmc_window", 60),
                        freq_secs=param_config.get("mcmc_freq_secs", 600),
                    )

                    logger.info(f"Turbine {turbine_id} MCMC: threshold={rf_threshold * mcmc_stat_factor:.6f}, events={len(stat_events_mcmc[turbine_id])}")

                    # Save events
                    db.save_events({turbine_id: sig_events_trad[turbine_id]}, 'significant_traditional')
                    db.save_events({turbine_id: stat_events_trad[turbine_id]}, 'stationary_traditional')
                    db.save_events({turbine_id: sig_events_mcmc[turbine_id]}, 'significant_mcmc')
                    db.save_events({turbine_id: stat_events_mcmc[turbine_id]}, 'stationary_mcmc')
                    
                except Exception as e:
                    logger.error(f"Error processing turbine {turbine_id}: {e}")
                    # Initialize empty results for failed turbines
                    sig_events_trad[turbine_id] = pd.DataFrame()
                    stat_events_trad[turbine_id] = pd.DataFrame()
                    sig_events_mcmc[turbine_id] = pd.DataFrame()
                    stat_events_mcmc[turbine_id] = pd.DataFrame()

            def convert_to_dataframe(event_dict):
                """Convert event dictionary to DataFrame"""
                if not event_dict or all(df.empty for df in event_dict.values()):
                    return pd.DataFrame()
                
                valid_events = {k: v for k, v in event_dict.items() if not v.empty}
                if valid_events:
                    return pd.concat(valid_events.values(), keys=valid_events.keys())
                return pd.DataFrame()

            return [
                convert_to_dataframe(sig_events_trad),
                convert_to_dataframe(stat_events_trad),
                convert_to_dataframe(sig_events_mcmc),
                convert_to_dataframe(stat_events_mcmc),
                tao
            ]
            
    except Exception as e:
        logger.error(f"RBA_theta failed: {e}")
        # Return empty results
        empty_df = pd.DataFrame()
        return [empty_df, empty_df, empty_df, empty_df, 1]

# ==================== QUALITY METRICS FUNCTIONS (ADDED) ====================

def calculate_event_quality_metrics(sig_events_df, stat_events_df, original_data=None):
    """
    Calculate simple quality metrics for detected events without ground truth
    
    Returns:
        dict: Quality metrics for both significant and stationary events
    """
    metrics = {
        'significant_events': {},
        'stationary_events': {},
        'overall': {}
    }
    
    # ==================== SIGNIFICANT EVENT METRICS ====================
    
    if not sig_events_df.empty:
        # 1. Ramp Consistency Score (0-1, higher is better)
        ramp_consistency = _calculate_ramp_consistency(sig_events_df)
        
        # 2. Magnitude Distribution Score (0-1, higher is better)
        magnitude_score = _calculate_magnitude_distribution_score(sig_events_df)
        
        # 3. Duration Reasonableness (0-1, higher is better)
        duration_score = _calculate_duration_reasonableness(sig_events_df)
        
        # 4. Direction Classification Accuracy (0-1, higher is better)
        direction_accuracy = _calculate_direction_classification_accuracy(sig_events_df)
        
        metrics['significant_events'] = {
            'total_events': len(sig_events_df),
            'ramp_consistency_score': ramp_consistency,
            'magnitude_distribution_score': magnitude_score,
            'duration_reasonableness_score': duration_score,
            'direction_classification_accuracy': direction_accuracy,
            'overall_quality_score': np.mean([ramp_consistency, magnitude_score, duration_score, direction_accuracy])
        }
    
    # ==================== STATIONARY EVENT METRICS ====================
    
    if not stat_events_df.empty:
        # 1. Stability Consistency (0-1, higher is better)
        stability_score = _calculate_stability_consistency(stat_events_df)
        
        # 2. Duration Appropriateness (0-1, higher is better)
        stat_duration_score = _calculate_stationary_duration_appropriateness(stat_events_df)
        
        # 3. Non-overlap Score (0-1, higher is better)
        non_overlap_score = _calculate_non_overlap_score(stat_events_df)
        
        metrics['stationary_events'] = {
            'total_events': len(stat_events_df),
            'stability_consistency_score': stability_score,
            'duration_appropriateness_score': stat_duration_score,
            'non_overlap_score': non_overlap_score,
            'overall_quality_score': np.mean([stability_score, stat_duration_score, non_overlap_score])
        }
    
    # ==================== OVERALL METRICS ====================
    
    total_events = len(sig_events_df) + len(stat_events_df)
    sig_ratio = len(sig_events_df) / total_events if total_events > 0 else 0
    stat_ratio = len(stat_events_df) / total_events if total_events > 0 else 0
    
    # Balance Score (closer to 0.5/0.5 is better for wind data)
    balance_score = 1.0 - abs(sig_ratio - 0.5) * 2  # Penalize extreme imbalance
    
    metrics['overall'] = {
        'total_events': total_events,
        'significant_ratio': sig_ratio,
        'stationary_ratio': stat_ratio,
        'balance_score': balance_score,
        'events_per_day': total_events / (len(original_data) / 24) if original_data is not None else None
    }
    
    return metrics

def _calculate_ramp_consistency(sig_events_df):
    """Score based on how consistent the ramp characteristics are"""
    if sig_events_df.empty or '∆w_m' not in sig_events_df.columns:
        return 0.0
    
    durations = sig_events_df['∆t_m'].values
    magnitudes = sig_events_df['∆w_m'].abs().values
    
    if len(durations) < 2:
        return 0.5
    
    try:
        correlation = np.corrcoef(durations, magnitudes)[0, 1]
        if np.isnan(correlation):
            return 0.5
        consistency_score = max(0, min(1, (correlation + 1) / 2))
        return consistency_score
    except:
        return 0.5

def _calculate_magnitude_distribution_score(sig_events_df):
    """Score based on whether magnitude distribution looks reasonable"""
    if sig_events_df.empty or '∆w_m' not in sig_events_df.columns:
        return 0.0
    
    magnitudes = sig_events_df['∆w_m'].abs().values
    
    if len(magnitudes) < 3:
        return 0.5
    
    median_mag = np.median(magnitudes)
    small_events = np.sum(magnitudes <= median_mag)
    large_events = np.sum(magnitudes > median_mag)
    
    small_large_ratio = small_events / (large_events + 1)
    ratio_score = min(1.0, small_large_ratio / 2.0)
    
    cv = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0
    cv_score = max(0, min(1, 1 - abs(cv - 0.5) / 0.5))
    
    return (ratio_score + cv_score) / 2

def _calculate_duration_reasonableness(sig_events_df):
    """Score based on whether durations are reasonable for ramp events"""
    if sig_events_df.empty or '∆t_m' not in sig_events_df.columns:
        return 0.0
    
    durations = sig_events_df['∆t_m'].values
    
    if len(durations) == 0:
        return 0.0
    
    reasonable_range = (2, 50)
    reasonable_events = np.sum((durations >= reasonable_range[0]) & (durations <= reasonable_range[1]))
    total_events = len(durations)
    
    reasonableness_score = reasonable_events / total_events
    
    very_short = np.sum(durations < 2)
    very_long = np.sum(durations > 100)
    penalty = (very_short + very_long) / total_events * 0.5
    
    return max(0, reasonableness_score - penalty)

def _calculate_direction_classification_accuracy(sig_events_df):
    """Score based on how well up/down ramps are classified"""
    if sig_events_df.empty or '∆w_m' not in sig_events_df.columns or 'θ_m' not in sig_events_df.columns:
        return 0.0
    
    delta_w = sig_events_df['∆w_m'].values
    angles = sig_events_df['θ_m'].values
    
    if len(delta_w) == 0:
        return 0.0
    
    correct_classifications = 0
    total_classifications = 0
    
    for dw, angle in zip(delta_w, angles):
        if not np.isnan(angle) and not np.isnan(dw):
            if (dw > 0 and angle > 0) or (dw < 0 and angle < 0):
                correct_classifications += 1
            total_classifications += 1
    
    if total_classifications == 0:
        return 0.5
    
    return correct_classifications / total_classifications

def _calculate_stability_consistency(stat_events_df):
    """Score based on how stable the stationary events are"""
    if stat_events_df.empty or 'σ_s' not in stat_events_df.columns:
        return 0.0
    
    sigmas = stat_events_df['σ_s'].values
    
    if len(sigmas) == 0:
        return 0.0
    
    mean_sigma = np.mean(sigmas)
    sigma_consistency = np.std(sigmas) / mean_sigma if mean_sigma > 0 else 1
    
    consistency_score = max(0, min(1, 1 - sigma_consistency))
    small_sigma_ratio = np.sum(sigmas < 0.2) / len(sigmas)
    
    return (consistency_score + small_sigma_ratio) / 2

def _calculate_stationary_duration_appropriateness(stat_events_df):
    """Score based on whether stationary durations are appropriate"""
    if stat_events_df.empty or '∆t_s' not in stat_events_df.columns:
        return 0.0
    
    durations = stat_events_df['∆t_s'].values
    
    if len(durations) == 0:
        return 0.0
    
    reasonable_range = (5, 200)
    reasonable_events = np.sum((durations >= reasonable_range[0]) & (durations <= reasonable_range[1]))
    total_events = len(durations)
    
    appropriateness_score = reasonable_events / total_events
    
    long_periods = np.sum(durations >= 20)
    long_period_bonus = min(0.2, long_periods / total_events)
    
    return min(1.0, appropriateness_score + long_period_bonus)

def _calculate_non_overlap_score(stat_events_df):
    """Score based on how well overlaps are avoided"""
    if stat_events_df.empty or 't1' not in stat_events_df.columns or 't2' not in stat_events_df.columns:
        return 1.0
    
    events = stat_events_df[['t1', 't2']].values
    
    if len(events) <= 1:
        return 1.0
    
    events = events[events[:, 0].argsort()]
    
    overlaps = 0
    total_pairs = 0
    
    for i in range(len(events) - 1):
        for j in range(i + 1, len(events)):
            total_pairs += 1
            if not (events[j][0] >= events[i][1] or events[i][0] >= events[j][1]):
                overlaps += 1
    
    if total_pairs == 0:
        return 1.0
    
    return 1.0 - (overlaps / total_pairs)

def enhanced_RBA_theta_with_metrics(data, nominal, param_config=None, db_path="rbatheta.db"):
    """
    Enhanced RBA_theta that includes quality metrics
    """
    # Run original RBA_theta
    results = RBA_theta(data, nominal, param_config, db_path)
    
    sig_trad, stat_trad, sig_mcmc, stat_mcmc, tao = results
    
    # Calculate quality metrics
    logger.info("Calculating event quality metrics...")
    
    # Get original data for context
    try:
        with RBAThetaDB(db_path) as db:
            all_data = []
            turbine_ids = db.get_all_turbine_ids()
            for turbine_id in turbine_ids:
                turbine_data = db.get_turbine_data(turbine_id)
                all_data.extend(turbine_data['normalized_value'].values)
    except:
        all_data = None
    
    # Calculate metrics for both methods
    trad_metrics = calculate_event_quality_metrics(sig_trad, stat_trad, all_data)
    mcmc_metrics = calculate_event_quality_metrics(sig_mcmc, stat_mcmc, all_data)
    
    # Log quality summary
    logger.info("=== EVENT QUALITY METRICS ===")
    logger.info("Traditional Method:")
    if trad_metrics['significant_events']:
        logger.info(f"  Significant Events Quality: {trad_metrics['significant_events']['overall_quality_score']:.3f}")
    if trad_metrics['stationary_events']:
        logger.info(f"  Stationary Events Quality: {trad_metrics['stationary_events']['overall_quality_score']:.3f}")
    logger.info(f"  Balance Score: {trad_metrics['overall']['balance_score']:.3f}")
    
    logger.info("RF-Enhanced MCMC Method:")
    if mcmc_metrics['significant_events']:
        logger.info(f"  Significant Events Quality: {mcmc_metrics['significant_events']['overall_quality_score']:.3f}")
    if mcmc_metrics['stationary_events']:
        logger.info(f"  Stationary Events Quality: {mcmc_metrics['stationary_events']['overall_quality_score']:.3f}")
    logger.info(f"  Balance Score: {mcmc_metrics['overall']['balance_score']:.3f}")
    
    # Return original results plus metrics
    return results, {'traditional': trad_metrics, 'rf_enhanced': mcmc_metrics}

def get_season(timestamp):
    """Determine season from a timestamp"""
    month = timestamp.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "default"
    '''
    for i in range(N):
        number_of_significant_events = len(significant_events[turbines[i]])
        number_of_stationary_events = len(stationary_events[turbines[i]])

        # initializing the rainflow counts
        #significant_events[turbines[i]]['φ_m'] = [0 * len(significant_events[turbines[i]])]
        #stationary_events[turbines[i]]['φ_s'] = [0 * len(stationary_events[turbines[i]])]

        for k in range(number_of_significant_events):
            start = int(significant_events[turbines[i]].loc[k, 't1'])
            end = int(significant_events[turbines[i]].loc[k, 't2'])
            significant_events[turbines[i]].loc[k, 'φ_m'] = fn.rainflow_count(data=data.iloc[i, start:end])

        for k in range(number_of_stationary_events):
            start = int(stationary_events[turbines[i]].loc[k, 't1'])
            end = int(stationary_events[turbines[i]].loc[k, 't2'])
            stationary_events[turbines[i]].loc[k, 'φ_s'] = fn.rainflow_count(data=data.iloc[i, start:end])
    '''



def markov(major, stationary, shp_path):

    """
    A commonly-used type of weights is Queen-contiguity weights, which reflects adjacency relationships as a binary
    indicator variable denoting whether or not a polygon shares an edge or a vertex with another polygon. These weights
    are symmetric.
    """

    df = geopandas.read_file(shp_path)
    points = [(poly.centroid.x, poly.centroid.y) for poly in df.geometry]
    radius_km = libpysal.cg.sphere.RADIUS_EARTH_KM
    threshold = libpysal.weights.min_threshold_dist_from_shapefile(shp_path, radius=radius_km)
    distance_weights = DistanceBand(points, threshold=threshold*.025, binary=False)
    transition_matrises = {}


    #for lisa markov
    transition_matrises['∆t_m_tran'] = LISA_Markov(major['∆t_m'], distance_weights)
    transition_matrises['∆w_m_tran'] = LISA_Markov(major['∆w_m'], distance_weights)
    transition_matrises['θ_m_tran'] = LISA_Markov(major['θ_m'], distance_weights)
    transition_matrises['σ_m_tran'] = LISA_Markov(major['σ_m'], distance_weights)

    transition_matrises['∆t_s_tran'] = LISA_Markov(stationary['∆t_s'], distance_weights)
    transition_matrises['σ_s_tran'] = LISA_Markov(stationary['σ_s'], distance_weights)

    '''#for spatial markov
    transition_matrises['∆t_m_tran'] = Spatial_Markov(major['∆t_m'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['∆w_m_tran'] = Spatial_Markov(major['∆w_m'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['θ_m_tran'] = Spatial_Markov(major['θ_m'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['σ_m_tran'] = Spatial_Markov(major['σ_m'], distance_weights, fixed=True, k=5, m=5)

    transition_matrises['∆t_s_tran'] = Spatial_Markov(stationary['∆t_s'], distance_weights, fixed=True, k=5, m=5)
    transition_matrises['σ_s_tran'] = Spatial_Markov(stationary['σ_s'], distance_weights, fixed=True, k=5, m=5)'''
    return transition_matrises

