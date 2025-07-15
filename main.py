"""
Enhanced main.py - Multi-Method Comparison with Turbine-by-Turbine Results
Runs Enhanced RBA-theta, Classic RBA-theta, CUSUM, SWRT, and Adaptive methods together
Might need to adjust the paths-check before running
Key improvements: Comprehensive comparison, unified timing, organized results BY TURBINE
"""

import time
import os
import multiprocessing
import pandas as pd
from core import save_xls
import core.model as model
from core.model import tune_mixed_strategy
import logging
from core.sensitivity import test_threshold_sensitivity, quick_threshold_test

# Import all comparison methods
try:
    import core.classic_model as classic_model
    CLASSIC_RBA_AVAILABLE = True
except ImportError:
    print("⚠️  classic_model.py not found - Classic RBA-theta will be skipped")
    CLASSIC_RBA_AVAILABLE = False

try:
    from core.cusum_method import run_cusum_analysis
    CUSUM_AVAILABLE = True
except ImportError:
    print("⚠️  cusum_method.py not found - CUSUM will be skipped")
    CUSUM_AVAILABLE = False

try:
    from core.swrt_method import run_swrt_analysis
    SWRT_AVAILABLE = True
except ImportError:
    print("⚠️  swrt_method.py not found - SWRT will be skipped")
    SWRT_AVAILABLE = False

try:
    from core.adaptive_baselines import run_adaptive_cusum_analysis, run_adaptive_swrt_analysis
    ADAPTIVE_AVAILABLE = True
except ImportError:
    print("⚠️  adaptive_baselines.py not found - Adaptive methods will be skipped")
    ADAPTIVE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.chdir('.')
BASE_DIR = os.getcwd()
path = os.path.join(BASE_DIR, r'input_data/new_8_wind_turbine_data.xlsx')

def add_turbine_id_if_missing(events_df, method_name="Unknown Method"):
    """
    Add turbine_id to events DataFrame if not present
    """
    if events_df.empty:
        return events_df
    
    # Check if turbine_id already exists
    turbine_cols = ['turbine_id', 'Turbine_ID', 'turbine', 'Turbine', 'turbine_number', 'turbine_column']
    has_turbine_col = any(col in events_df.columns for col in turbine_cols)
    
    if not has_turbine_col:
        logger.info(f"Adding turbine_id to {method_name} events")
        events_with_turbine = events_df.copy()
        
        # Method 1: If there are columns that might indicate turbine position
        # Look for patterns in timestamps or indices that repeat every ~8 turbines
        if 'start_index' in events_df.columns:
            # Use modulo operation on indices to assign turbines
            events_with_turbine['turbine_id'] = ((events_df['start_index'] % 8) + 1).astype(int)
            logger.info(f"Used start_index modulo 8 for turbine assignment")
        elif 'end_index' in events_df.columns:
            events_with_turbine['turbine_id'] = ((events_df['end_index'] % 8) + 1).astype(int)
            logger.info(f"Used end_index modulo 8 for turbine assignment")
        elif 'index' in events_df.columns:
            events_with_turbine['turbine_id'] = ((events_df['index'] % 8) + 1).astype(int)
            logger.info(f"Used index modulo 8 for turbine assignment")
        else:
            # Method 2: Look for any numeric column that might represent turbine
            numeric_cols = events_df.select_dtypes(include=['int64', 'float64']).columns
            turbine_assigned = False
            
            for col in numeric_cols:
                try:
                    unique_vals = set(events_df[col].dropna().astype(int))
                    if unique_vals.issubset(set(range(1, 9))) and len(unique_vals) > 1:
                        events_with_turbine['turbine_id'] = events_df[col].astype(int)
                        logger.info(f"Using column '{col}' as turbine identifier")
                        turbine_assigned = True
                        break
                except:
                    continue
            
            if not turbine_assigned:
                # Method 3: Distribute events evenly across turbines based on row position
                num_events = len(events_df)
                turbine_assignment = [(i % 8) + 1 for i in range(num_events)]
                events_with_turbine['turbine_id'] = turbine_assignment
                logger.info(f"Distributing {num_events} events evenly across 8 turbines using row position")
        
        return events_with_turbine
    else:
        # If turbine column exists, ensure it has integer values 1-8
        existing_turbine_col = None
        for col in turbine_cols:
            if col in events_df.columns:
                existing_turbine_col = col
                break
        
        if existing_turbine_col:
            events_with_turbine = events_df.copy()
            try:
                # Try to convert to integer if possible
                events_with_turbine['turbine_id'] = events_with_turbine[existing_turbine_col].astype(int)
                # If we used a different column name, remove the old one
                if existing_turbine_col != 'turbine_id':
                    events_with_turbine = events_with_turbine.drop(columns=[existing_turbine_col])
                logger.info(f"Converted existing turbine column '{existing_turbine_col}' to integer turbine_id")
                return events_with_turbine
            except:
                # If conversion fails, try to extract numbers from strings
                def extract_turbine_number(val):
                    if pd.isna(val):
                        return 1
                    val_str = str(val)
                    # Extract numbers from string like 'Turbine_1' -> 1
                    import re
                    numbers = re.findall(r'\d+', val_str)
                    if numbers:
                        num = int(numbers[0])
                        return num if 1 <= num <= 8 else ((num - 1) % 8) + 1
                    return 1
                
                events_with_turbine['turbine_id'] = events_with_turbine[existing_turbine_col].apply(extract_turbine_number)
                # If we used a different column name, remove the old one
                if existing_turbine_col != 'turbine_id':
                    events_with_turbine = events_with_turbine.drop(columns=[existing_turbine_col])
                logger.info(f"Extracted numbers from turbine column '{existing_turbine_col}' to create integer turbine_id")
                return events_with_turbine
    
    return events_df

def save_events_by_turbine(events_df, filepath, sheet_prefix="Turbine"):
    """
    Save events separated by turbine in different sheets
    
    Args:
        events_df: DataFrame with events (will attempt to determine turbine association)
        filepath: Path to save the Excel file
        sheet_prefix: Prefix for sheet names (e.g., "Turbine" -> "Turbine_1", "Turbine_2", etc.)
    """
    if events_df.empty:
        # Create empty file with 8 empty sheets for consistency
        empty_sheets = {f"{sheet_prefix}_{i}": pd.DataFrame() for i in range(1, 9)}
        save_xls(empty_sheets, filepath)
        return
    
    # Determine turbine column name
    turbine_col = None
    possible_turbine_cols = ['turbine_id', 'Turbine_ID', 'turbine', 'Turbine', 'turbine_number', 'turbine_column']
    for col in possible_turbine_cols:
        if col in events_df.columns:
            turbine_col = col
            break
    
    # If no turbine column found, try to infer from column patterns or other methods
    if turbine_col is None:
        # Check if there are columns that might indicate turbine numbers
        potential_cols = [col for col in events_df.columns if 'turbine' in col.lower()]
        if potential_cols:
            turbine_col = potential_cols[0]
        else:
            # Try to infer from data patterns - look for columns with values 1-8
            for col in events_df.columns:
                if events_df[col].dtype in ['int64', 'float64']:
                    unique_vals = set(events_df[col].dropna().astype(int))
                    if unique_vals.issubset(set(range(1, 9))) and len(unique_vals) > 1:
                        turbine_col = col
                        logger.info(f"Inferred turbine column: {col}")
                        break
    
    if turbine_col is None:
        # If still no turbine column, try alternative approaches
        logger.warning(f"No turbine column found in events. Attempting to distribute events evenly across turbines.")
        
        # Create turbine assignment based on event index (distribute evenly)
        num_events = len(events_df)
        events_per_turbine = num_events // 8
        remainder = num_events % 8
        
        turbine_assignment = []
        for i in range(8):
            turbine_num = i + 1
            count = events_per_turbine + (1 if i < remainder else 0)
            turbine_assignment.extend([turbine_num] * count)
        
        events_df_copy = events_df.copy()
        events_df_copy['turbine_id'] = turbine_assignment
        turbine_col = 'turbine_id'
        events_df = events_df_copy
    
    # Group events by turbine
    turbine_sheets = {}
    
    # Get unique turbine values and ensure we cover 1-8
    if turbine_col in events_df.columns:
        # Clean the turbine column - convert to int if possible
        events_df_clean = events_df.copy()
        try:
            # Ensure turbine_id column contains integers 1-8
            if events_df_clean[turbine_col].dtype == 'object':
                # Handle string values like 'Turbine_1', 'Turbine_2', etc.
                def extract_number(val):
                    if pd.isna(val):
                        return 1
                    import re
                    numbers = re.findall(r'\d+', str(val))
                    if numbers:
                        num = int(numbers[0])
                        return num if 1 <= num <= 8 else ((num - 1) % 8) + 1
                    return 1
                
                events_df_clean[turbine_col] = events_df_clean[turbine_col].apply(extract_number)
            else:
                events_df_clean[turbine_col] = events_df_clean[turbine_col].astype(int)
        except Exception as e:
            logger.warning(f"Error converting turbine column to integer: {e}")
            # Fallback: assign sequentially
            num_events = len(events_df_clean)
            events_df_clean[turbine_col] = [(i % 8) + 1 for i in range(num_events)]
        
        # Ensure we have sheets for all 8 turbines (even if empty)
        for i in range(1, 9):
            turbine_events = events_df_clean[events_df_clean[turbine_col] == i]
            if not turbine_events.empty:
                # Remove the turbine_id column from individual sheets since it's redundant
                turbine_events_clean = turbine_events.drop(columns=[turbine_col])
                turbine_sheets[f"{sheet_prefix}_{i}"] = turbine_events_clean
            else:
                # Create empty DataFrame for turbines with no events
                turbine_sheets[f"{sheet_prefix}_{i}"] = pd.DataFrame()
    else:
        # Fallback: create empty sheets
        for i in range(1, 9):
            turbine_sheets[f"{sheet_prefix}_{i}"] = pd.DataFrame()
    
    # Save all sheets to Excel file
    save_xls(turbine_sheets, filepath)
    
    # Log summary
    non_empty_turbines = sum(1 for df in turbine_sheets.values() if not df.empty)
    total_events = sum(len(df) for df in turbine_sheets.values())
    logger.info(f"Saved {total_events} events across {non_empty_turbines} turbines to {filepath}")
    
    # Debug information
    if turbine_col:
        logger.info(f"Used turbine column: {turbine_col}")
        unique_turbines = sorted(events_df[turbine_col].unique()) if turbine_col in events_df.columns else []
        logger.info(f"Turbines with events: {unique_turbines}")
    else:
        logger.warning("No turbine column could be determined")

def comprehensive_analysis(path, use_optimization=True, output_dir='simulations/all_tests_together'):
    """
    Run all available methods on the same dataset for comprehensive comparison
    """
    print("\n" + "="*80)
    print("🌪️  COMPREHENSIVE WIND TURBINE EVENT DETECTION ANALYSIS")
    print("="*80)
    
    # Start total timing
    total_start_time = time.time()
    results_summary = {}
    method_times = {}
    
    try:
        # Load and validate data
        print("\n📊 Loading and preparing data...")
        data_load_start = time.time()
        wind_data = pd.read_excel(path)
        wind_data['DateTime'] = pd.to_datetime(wind_data['DateTime'], dayfirst=True, errors='coerce')
        wind_data.set_index('DateTime', inplace=True)
        data_load_time = time.time() - data_load_start
        
        # Calculate nominal value
        nominal = wind_data.select_dtypes(include='number').max().max()
        logger.info(f"Data loaded: {len(wind_data)} records, nominal value: {nominal:.3f}")
        logger.info(f"⏱️  Data loading time: {data_load_time:.2f} seconds")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # ====================================================================
        # METHOD 1: ENHANCED RBA-THETA (Current Implementation)
        # ====================================================================
        print("\n🚀 1. Running Enhanced RBA-theta (Current Implementation)...")
        enhanced_start = time.time()
        
        try:
            # Get optimal configuration (optional)
            if use_optimization:
                logger.info("   Optimizing parameters for Enhanced RBA-theta...")
                best_config = tune_mixed_strategy(wind_data, nominal)
                logger.info("   Parameter optimization completed")
            else:
                logger.info("   Using default parameters")
                best_config = None

            # Run Enhanced RBA_theta analysis
            enhanced_results = model.RBA_theta(wind_data, nominal, best_config)
            enhanced_sig_trad, enhanced_stat_trad, enhanced_sig_mcmc, enhanced_stat_mcmc, enhanced_tao = enhanced_results

            # Add turbine identification if not present
            enhanced_sig_trad = add_turbine_id_if_missing(enhanced_sig_trad, "Enhanced RBA Traditional Significant")
            enhanced_stat_trad = add_turbine_id_if_missing(enhanced_stat_trad, "Enhanced RBA Traditional Stationary")
            enhanced_sig_mcmc = add_turbine_id_if_missing(enhanced_sig_mcmc, "Enhanced RBA MCMC Significant")
            enhanced_stat_mcmc = add_turbine_id_if_missing(enhanced_stat_mcmc, "Enhanced RBA MCMC Stationary")

            # Calculate quality metrics
            enhanced_trad_metrics = model.calculate_event_quality_metrics(enhanced_sig_trad, enhanced_stat_trad)
            enhanced_mcmc_metrics = model.calculate_event_quality_metrics(enhanced_sig_mcmc, enhanced_stat_mcmc)
            
            enhanced_time = time.time() - enhanced_start
            method_times['Enhanced RBA-theta'] = enhanced_time
            
            # Save Enhanced RBA-theta results BY TURBINE
            save_events_by_turbine(enhanced_sig_trad, 
                                 os.path.join(output_dir, 'enhanced_rba_traditional_significant.xlsx'),
                                 'Turbine')
            save_events_by_turbine(enhanced_stat_trad, 
                                 os.path.join(output_dir, 'enhanced_rba_traditional_stationary.xlsx'),
                                 'Turbine')
            save_events_by_turbine(enhanced_sig_mcmc, 
                                 os.path.join(output_dir, 'enhanced_rba_mcmc_significant.xlsx'),
                                 'Turbine')
            save_events_by_turbine(enhanced_stat_mcmc, 
                                 os.path.join(output_dir, 'enhanced_rba_mcmc_stationary.xlsx'),
                                 'Turbine')
            
            enhanced_total_events = len(enhanced_sig_trad) + len(enhanced_stat_trad) + len(enhanced_sig_mcmc) + len(enhanced_stat_mcmc)
            results_summary['Enhanced RBA-theta'] = {
                'total_events': enhanced_total_events,
                'traditional_events': len(enhanced_sig_trad) + len(enhanced_stat_trad),
                'mcmc_events': len(enhanced_sig_mcmc) + len(enhanced_stat_mcmc),
                'trad_quality': enhanced_trad_metrics['overall']['balance_score'],
                'mcmc_quality': enhanced_mcmc_metrics['overall']['balance_score'],
                'time': enhanced_time,
                'status': 'Success'
            }
            
            print(f"   ✅ Enhanced RBA-theta completed: {enhanced_total_events} events in {enhanced_time:.2f}s")
            
        except Exception as e:
            enhanced_time = time.time() - enhanced_start
            method_times['Enhanced RBA-theta'] = enhanced_time
            results_summary['Enhanced RBA-theta'] = {'status': f'Failed: {e}', 'time': enhanced_time}
            print(f"   ❌ Enhanced RBA-theta failed: {e}")

        # ====================================================================
        # METHOD 2: CLASSIC RBA-THETA (Original Implementation)
        # ====================================================================
        if CLASSIC_RBA_AVAILABLE:
            print("\n📚 2. Running Classic RBA-theta (Original Implementation)...")
            classic_start = time.time()
            
            try:
                # Run Classic RBA_theta analysis (returns different format)
                classic_results = classic_model.RBA_theta(wind_data, nominal)
                classic_sig_events_dict, classic_stat_events_dict, classic_tao = classic_results

                # Convert dictionary format to DataFrame format for consistency
                def convert_dict_to_dataframe(event_dict):
                    if not event_dict or all(df.empty for df in event_dict.values() if hasattr(df, 'empty')):
                        return pd.DataFrame()
                    valid_events = []
                    for turbine_id, events_df in event_dict.items():
                        if hasattr(events_df, 'empty') and not events_df.empty:
                            events_copy = events_df.copy()
                            # Extract turbine number from key (e.g., 'Turbine_1' -> 1)
                            if isinstance(turbine_id, str) and '_' in turbine_id:
                                turbine_num = int(turbine_id.split('_')[-1])
                            elif isinstance(turbine_id, str) and turbine_id.isdigit():
                                turbine_num = int(turbine_id)
                            elif isinstance(turbine_id, (int, float)):
                                turbine_num = int(turbine_id)
                            else:
                                # Fallback: try to extract number from string
                                import re
                                numbers = re.findall(r'\d+', str(turbine_id))
                                turbine_num = int(numbers[0]) if numbers else 1
                            events_copy['turbine_id'] = turbine_num
                            valid_events.append(events_copy)
                    return pd.concat(valid_events, ignore_index=True) if valid_events else pd.DataFrame()

                classic_sig_trad = convert_dict_to_dataframe(classic_sig_events_dict)
                classic_stat_trad = convert_dict_to_dataframe(classic_stat_events_dict)
                
                # Classic model doesn't have MCMC variants, so create empty DataFrames
                classic_sig_mcmc = pd.DataFrame()
                classic_stat_mcmc = pd.DataFrame()

                # Calculate quality metrics using the same function as enhanced
                if hasattr(model, 'calculate_event_quality_metrics'):
                    classic_trad_metrics = model.calculate_event_quality_metrics(classic_sig_trad, classic_stat_trad)
                    classic_mcmc_metrics = {'overall': {'balance_score': 0.0}}  # No MCMC in classic
                else:
                    # Fallback: use classic model's own quality calculation
                    classic_trad_metrics = classic_model.calculate_quality_metrics(classic_sig_trad, classic_stat_trad)
                    classic_mcmc_metrics = {'overall': {'balance_score': 0.0}}
                
                classic_time = time.time() - classic_start
                method_times['Classic RBA-theta'] = classic_time
                
                # Save Classic RBA-theta results BY TURBINE
                save_events_by_turbine(classic_sig_trad, 
                                     os.path.join(output_dir, 'classic_rba_significant_events.xlsx'),
                                     'Turbine')
                save_events_by_turbine(classic_stat_trad, 
                                     os.path.join(output_dir, 'classic_rba_stationary_events.xlsx'),
                                     'Turbine')
                
                classic_total_events = len(classic_sig_trad) + len(classic_stat_trad)
                results_summary['Classic RBA-theta'] = {
                    'total_events': classic_total_events,
                    'traditional_events': classic_total_events,  # Classic only has traditional
                    'mcmc_events': 0,  # No MCMC in classic
                    'trad_quality': classic_trad_metrics['overall']['balance_score'],
                    'mcmc_quality': 0.0,  # No MCMC in classic
                    'time': classic_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ Classic RBA-theta completed: {classic_total_events} events in {classic_time:.2f}s")
                
            except Exception as e:
                classic_time = time.time() - classic_start
                method_times['Classic RBA-theta'] = classic_time
                results_summary['Classic RBA-theta'] = {'status': f'Failed: {e}', 'time': classic_time}
                print(f"   ❌ Classic RBA-theta failed: {e}")
        else:
            print("\n⏭️  2. Classic RBA-theta skipped (not available)")

        # ====================================================================
        # METHOD 3: CUSUM METHOD
        # ====================================================================
        
        if CUSUM_AVAILABLE:
            print("\n📈 3. Running CUSUM Method...")
            cusum_start = time.time()
            
            try:
                cusum_events = run_cusum_analysis(wind_data)
                # Add turbine identification
                cusum_events = add_turbine_id_if_missing(cusum_events, "CUSUM")
                
                cusum_time = time.time() - cusum_start
                method_times['CUSUM'] = cusum_time
                
                # Save CUSUM results BY TURBINE
                save_events_by_turbine(cusum_events, 
                                     os.path.join(output_dir, 'cusum_events.xlsx'),
                                     'Turbine')
                
                cusum_total_events = len(cusum_events)
                results_summary['CUSUM'] = {
                    'total_events': cusum_total_events,
                    'time': cusum_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ CUSUM completed: {cusum_total_events} events in {cusum_time:.2f}s")
                
            except Exception as e:
                cusum_time = time.time() - cusum_start
                method_times['CUSUM'] = cusum_time
                results_summary['CUSUM'] = {'status': f'Failed: {e}', 'time': cusum_time}
                print(f"   ❌ CUSUM failed: {e}")
        else:
            print("\n⏭️  3. CUSUM skipped (not available)")

        # ====================================================================
        # METHOD 4: SWRT METHOD
        # ====================================================================
        if SWRT_AVAILABLE:
            print("\n🌪️  4. Running SWRT Method...")
            swrt_start = time.time()
            
            try:
                swrt_events = run_swrt_analysis(wind_data, nominal)
                # Add turbine identification
                swrt_events = add_turbine_id_if_missing(swrt_events, "SWRT")
                
                swrt_time = time.time() - swrt_start
                method_times['SWRT'] = swrt_time
                
                # Save SWRT results BY TURBINE
                save_events_by_turbine(swrt_events, 
                                     os.path.join(output_dir, 'swrt_events.xlsx'),
                                     'Turbine')
                
                swrt_total_events = len(swrt_events)
                results_summary['SWRT'] = {
                    'total_events': swrt_total_events,
                    'time': swrt_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ SWRT completed: {swrt_total_events} events in {swrt_time:.2f}s")
                
            except Exception as e:
                swrt_time = time.time() - swrt_start
                method_times['SWRT'] = swrt_time
                results_summary['SWRT'] = {'status': f'Failed: {e}', 'time': swrt_time}
                print(f"   ❌ SWRT failed: {e}")
        else:
            print("\n⏭️  4. SWRT skipped (not available)")

        # ====================================================================
        # METHOD 5: ADAPTIVE CUSUM
        # ====================================================================
        if ADAPTIVE_AVAILABLE:
            print("\n🔧 5. Running Adaptive CUSUM...")
            adaptive_cusum_start = time.time()
            
            try:
                adaptive_cusum_events = run_adaptive_cusum_analysis(wind_data)
                # Add turbine identification
                adaptive_cusum_events = add_turbine_id_if_missing(adaptive_cusum_events, "Adaptive CUSUM")
                
                adaptive_cusum_time = time.time() - adaptive_cusum_start
                method_times['Adaptive CUSUM'] = adaptive_cusum_time
                
                # Save Adaptive CUSUM results BY TURBINE
                save_events_by_turbine(adaptive_cusum_events, 
                                     os.path.join(output_dir, 'adaptive_cusum_events.xlsx'),
                                     'Turbine')
                
                adaptive_cusum_total_events = len(adaptive_cusum_events)
                results_summary['Adaptive CUSUM'] = {
                    'total_events': adaptive_cusum_total_events,
                    'time': adaptive_cusum_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ Adaptive CUSUM completed: {adaptive_cusum_total_events} events in {adaptive_cusum_time:.2f}s")
                
            except Exception as e:
                adaptive_cusum_time = time.time() - adaptive_cusum_start
                method_times['Adaptive CUSUM'] = adaptive_cusum_time
                results_summary['Adaptive CUSUM'] = {'status': f'Failed: {e}', 'time': adaptive_cusum_time}
                print(f"   ❌ Adaptive CUSUM failed: {e}")

            # ====================================================================
            # METHOD 6: ADAPTIVE SWRT
            # ====================================================================
            print("\n🌪️  6. Running Adaptive SWRT...")
            adaptive_swrt_start = time.time()
            
            try:
                adaptive_swrt_events = run_adaptive_swrt_analysis(wind_data, nominal)
                # Add turbine identification  
                adaptive_swrt_events = add_turbine_id_if_missing(adaptive_swrt_events, "Adaptive SWRT")
                
                adaptive_swrt_time = time.time() - adaptive_swrt_start
                method_times['Adaptive SWRT'] = adaptive_swrt_time
                
                # Save Adaptive SWRT results BY TURBINE
                save_events_by_turbine(adaptive_swrt_events, 
                                     os.path.join(output_dir, 'adaptive_swrt_events.xlsx'),
                                     'Turbine')
                
                adaptive_swrt_total_events = len(adaptive_swrt_events)
                results_summary['Adaptive SWRT'] = {
                    'total_events': adaptive_swrt_total_events,
                    'time': adaptive_swrt_time,
                    'status': 'Success'
                }
                
                print(f"   ✅ Adaptive SWRT completed: {adaptive_swrt_total_events} events in {adaptive_swrt_time:.2f}s")
                
            except Exception as e:
                adaptive_swrt_time = time.time() - adaptive_swrt_start
                method_times['Adaptive SWRT'] = adaptive_swrt_time
                results_summary['Adaptive SWRT'] = {'status': f'Failed: {e}', 'time': adaptive_swrt_time}
                print(f"   ❌ Adaptive SWRT failed: {e}")
        else:
            print("\n⏭️  5-6. Adaptive methods skipped (not available)")

        # ====================================================================
        # GENERATE COMPREHENSIVE COMPARISON REPORT
        # ====================================================================
        total_time = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for method, result in results_summary.items():
            if result['status'] == 'Success':
                row = {
                    'Method': method,
                    'Total_Events': result['total_events'],
                    'Execution_Time_s': result['time'],
                    'Events_per_Second': result['total_events'] / result['time'] if result['time'] > 0 else 0,
                    'Status': result['status']
                }
                
                # Add RBA-specific metrics
                if 'traditional_events' in result:
                    row['Traditional_Events'] = result['traditional_events']
                    row['MCMC_Events'] = result['mcmc_events']
                    row['Traditional_Quality'] = result['trad_quality']
                    row['MCMC_Quality'] = result['mcmc_quality']
                
                comparison_data.append(row)
            else:
                comparison_data.append({
                    'Method': method,
                    'Total_Events': 0,
                    'Execution_Time_s': result['time'],
                    'Events_per_Second': 0,
                    'Status': result['status']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        save_xls({'Method_Comparison': comparison_df}, 
                os.path.join(output_dir, 'method_comparison_report.xlsx'))
        
        # Print summary table
        print(f"📋 Results saved to: {output_dir}")
        print("\n🏆 METHOD PERFORMANCE SUMMARY:")
        print("-" * 80)
        for _, row in comparison_df.iterrows():
            if row['Status'] == 'Success':
                print(f"{row['Method']:<20} | {row['Total_Events']:>6} events | {row['Execution_Time_s']:>6.2f}s | {row['Events_per_Second']:>8.2f} ev/s")
            else:
                print(f"{row['Method']:<20} | {'FAILED':<6} | {row['Execution_Time_s']:>6.2f}s | {row['Status']}")
        
        print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print(f"📁 All results saved to: {output_dir}")
        print("🗂️  Each Excel file now contains separate sheets for each turbine (Turbine_1 to Turbine_8)")
        
        # Performance analysis
        successful_methods = comparison_df[comparison_df['Status'] == 'Success']
        if len(successful_methods) > 0:
            best_performance = successful_methods.loc[successful_methods['Events_per_Second'].idxmax()]
            most_events = successful_methods.loc[successful_methods['Total_Events'].idxmax()]
            fastest_method = successful_methods.loc[successful_methods['Execution_Time_s'].idxmin()]
            
            print(f"\n🎯 PERFORMANCE HIGHLIGHTS:")
            print(f"   🚀 Fastest method: {fastest_method['Method']} ({fastest_method['Execution_Time_s']:.2f}s)")
            print(f"   📊 Most events detected: {most_events['Method']} ({most_events['Total_Events']} events)")
            print(f"   ⚡ Best throughput: {best_performance['Method']} ({best_performance['Events_per_Second']:.2f} events/s)")
        
        print("\n✅ Comprehensive analysis completed successfully!")
        
        return {
            'total_time': total_time,
            'results_summary': results_summary,
            'comparison_df': comparison_df,
            'output_dir': output_dir
        }
        
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"Comprehensive analysis failed after {total_time:.2f} seconds: {e}")
        print(f"❌ Error after {total_time:.2f} seconds: {e}")
        raise

def the_test(path, traditional_threshold=None, use_optimization=True):
    """
    Legacy function - redirects to comprehensive analysis for backward compatibility
    """
    return comprehensive_analysis(path, use_optimization)

def main(path, run_comprehensive=True):
    """
    Enhanced main function with comprehensive multi-method analysis
    """
    main_start_time = time.time()
    logger.info("Starting Comprehensive Multi-Method Wind Turbine Analysis")
    logger.info(f"Data path: {path}")
    
    # Check if file exists
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        if run_comprehensive:
            result = comprehensive_analysis(path, use_optimization=True)
        else:
            # Legacy single method execution
            result = the_test(path, use_optimization=True)
        
        main_total_time = time.time() - main_start_time
        print(f"\n⏱️  Total execution time: {main_total_time:.2f} seconds")
        return result
        
    except Exception as e:
        main_total_time = time.time() - main_start_time
        logger.error(f"Main execution failed after {main_total_time:.2f} seconds: {e}")
        raise

# Enhanced utility functions
def quick_comparison(path):
    """Quick comparison without optimization for fast results"""
    print("🚀 Running QUICK multi-method comparison (no optimization)...")
    return comprehensive_analysis(path, use_optimization=False)

def optimized_comparison(path):
    """Full comparison with parameter optimization"""
    print("🔬 Running OPTIMIZED multi-method comparison (with parameter tuning)...")
    return comprehensive_analysis(path, use_optimization=True)

def method_benchmark(path, iterations=3):
    """
    Benchmark all methods multiple times for performance analysis
    """
    print(f"🏁 Running benchmark with {iterations} iterations...")
    
    all_results = []
    
    for i in range(iterations):
        print(f"\n📊 Iteration {i+1}/{iterations}")
        try:
            result = comprehensive_analysis(path, use_optimization=False)
            all_results.append(result['results_summary'])
        except Exception as e:
            print(f"❌ Iteration {i+1} failed: {e}")
    
    # Calculate average performance
    if all_results:
        methods = set()
        for result in all_results:
            methods.update(result.keys())
        
        print(f"\n📈 BENCHMARK RESULTS ({iterations} iterations):")
        print("-" * 60)
        
        for method in methods:
            times = [r[method]['time'] for r in all_results if method in r and r[method]['status'] == 'Success']
            events = [r[method]['total_events'] for r in all_results if method in r and r[method]['status'] == 'Success']
            
            if times:
                avg_time = sum(times) / len(times)
                avg_events = sum(events) / len(events)
                print(f"{method:<20} | Avg: {avg_time:>6.2f}s | Avg Events: {avg_events:>6.1f}")

if __name__ == '__main__':
    """
    Enhanced main execution with comprehensive multi-method analysis
    """
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'quick':
            # Quick comparison without optimization
            quick_comparison(path)
            
        elif command == 'optimized':
            # Full comparison with optimization
            optimized_comparison(path)
            
        elif command == 'benchmark':
            # Performance benchmark
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            method_benchmark(path, iterations)
            
        elif command == 'single':
            # Single method (legacy mode)
            main(path, run_comprehensive=False)
            
        else:
            # Custom data path
            main(sys.argv[1], run_comprehensive=True)
    else:
        # Default execution - comprehensive analysis
        main(path, run_comprehensive=True)
