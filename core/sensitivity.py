import time
import pandas as pd
import core.model as model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_threshold_sensitivity(data, nominal, base_config, threshold_multipliers = [0.01, 0.1, 1.0, 10.0, 20.0, 50.0, 100.0, 200.0]):
    """
    Test sensitivity to threshold changes
    
    Args:
        data: Wind turbine data
        nominal: Nominal power value
        base_config: Your optimized configuration
        threshold_multipliers: List of multipliers to test (1.0 = baseline)
    """
    
    results = []
    
    print("="*80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Testing {len(threshold_multipliers)} different threshold levels...")
    
    for i, multiplier in enumerate(threshold_multipliers):
        print(f"\n🔍 Test {i+1}/{len(threshold_multipliers)}: Threshold Multiplier = {multiplier:.1f}x")
        
        # Create modified config
        test_config = base_config.copy()
        test_config['sig_event_factor'] = base_config['sig_event_factor'] * multiplier
        test_config['stat_event_factor'] = base_config['stat_event_factor'] * multiplier
        
        print(f"   Significant threshold: {test_config['sig_event_factor']:.6f}")
        print(f"   Stationary threshold:  {test_config['stat_event_factor']:.6f}")
        
        # Run analysis with timing
        start_time = time.time()
        
        try:
            # Run RBA_theta with modified thresholds
            rba_results = model.RBA_theta(data, nominal, test_config)
            sig_trad, stat_trad, sig_mcmc, stat_mcmc, tao = rba_results
            
            # Calculate metrics
            trad_metrics = model.calculate_event_quality_metrics(sig_trad, stat_trad)
            mcmc_metrics = model.calculate_event_quality_metrics(sig_mcmc, stat_mcmc)
            
            analysis_time = time.time() - start_time
            
            # Count events
            total_events = len(sig_trad) + len(stat_trad) + len(sig_mcmc) + len(stat_mcmc)
            trad_events = len(sig_trad) + len(stat_trad)
            mcmc_events = len(sig_mcmc) + len(stat_mcmc)
            
            # Store results
            result = {
                'multiplier': multiplier,
                'sig_threshold': test_config['sig_event_factor'],
                'stat_threshold': test_config['stat_event_factor'],
                'total_events': total_events,
                'traditional_events': trad_events,
                'mcmc_events': mcmc_events,
                'trad_sig_quality': trad_metrics['significant_events']['overall_quality_score'] if trad_metrics['significant_events'] else 0,
                'trad_stat_quality': trad_metrics['stationary_events']['overall_quality_score'] if trad_metrics['stationary_events'] else 0,
                'trad_balance': trad_metrics['overall']['balance_score'],
                'mcmc_sig_quality': mcmc_metrics['significant_events']['overall_quality_score'] if mcmc_metrics['significant_events'] else 0,
                'mcmc_stat_quality': mcmc_metrics['stationary_events']['overall_quality_score'] if mcmc_metrics['stationary_events'] else 0,
                'mcmc_balance': mcmc_metrics['overall']['balance_score'],
                'analysis_time': analysis_time
            }
            results.append(result)
            
            print(f"   ✅ Total Events: {total_events} (Trad: {trad_events}, MCMC: {mcmc_events})")
            print(f"   ⏱️  Analysis Time: {analysis_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Add failed result
            result = {
                'multiplier': multiplier,
                'sig_threshold': test_config['sig_event_factor'],
                'stat_threshold': test_config['stat_event_factor'],
                'total_events': 0,
                'traditional_events': 0,
                'mcmc_events': 0,
                'error': str(e)
            }
            results.append(result)
    
    # Create summary report
    print_sensitivity_report(results, base_config)
    
    return results

def print_sensitivity_report(results, base_config):
    """
    Print detailed sensitivity analysis report
    """
    print("\n" + "="*100)
    print("THRESHOLD SENSITIVITY ANALYSIS REPORT")
    print("="*100)
    
    # Find baseline (multiplier = 1.0)
    baseline = next((r for r in results if r['multiplier'] == 1.0), results[0])
    
    print(f"\n📊 EVENT COUNT SENSITIVITY:")
    print("-" * 70)
    print(f"{'Multiplier':<12} {'Sig Thresh':<12} {'Total Events':<12} {'Change':<10} {'Trad':<8} {'MCMC':<8}")
    print("-" * 70)
    
    for result in results:
        if 'error' not in result:
            multiplier = result['multiplier']
            sig_thresh = result['sig_threshold']
            total = result['total_events']
            change = ((total - baseline['total_events']) / baseline['total_events'] * 100) if baseline['total_events'] > 0 else 0
            trad = result['traditional_events']
            mcmc = result['mcmc_events']
            
            print(f"{multiplier:<12.1f} {sig_thresh:<12.6f} {total:<12d} {change:+6.1f}%   {trad:<8d} {mcmc:<8d}")
    
    print(f"\n🎯 QUALITY SENSITIVITY (Traditional Method):")
    print("-" * 80)
    print(f"{'Multiplier':<12} {'Sig Quality':<12} {'Stat Quality':<12} {'Balance':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in results:
        if 'error' not in result:
            multiplier = result['multiplier']
            sig_qual = result.get('trad_sig_quality', 0)
            stat_qual = result.get('trad_stat_quality', 0)
            balance = result.get('trad_balance', 0)
            time_val = result.get('analysis_time', 0)
            
            print(f"{multiplier:<12.1f} {sig_qual:<12.3f} {stat_qual:<12.3f} {balance:<12.3f} {time_val:<10.1f}")
    
    print(f"\n🚀 QUALITY SENSITIVITY (MCMC Method):")
    print("-" * 80)
    print(f"{'Multiplier':<12} {'Sig Quality':<12} {'Stat Quality':<12} {'Balance':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in results:
        if 'error' not in result:
            multiplier = result['multiplier']
            sig_qual = result.get('mcmc_sig_quality', 0)
            stat_qual = result.get('mcmc_stat_quality', 0)
            balance = result.get('mcmc_balance', 0)
            time_val = result.get('analysis_time', 0)
            
            print(f"{multiplier:<12.1f} {sig_qual:<12.3f} {stat_qual:<12.3f} {balance:<12.3f} {time_val:<10.1f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 50)
    
    if len(results) > 1:
        max_events = max([r['total_events'] for r in results if 'error' not in r])
        min_events = min([r['total_events'] for r in results if 'error' not in r])
        
        max_result = next(r for r in results if r.get('total_events') == max_events)
        min_result = next(r for r in results if r.get('total_events') == min_events)
        
        print(f"• Most events ({max_events}): Multiplier {max_result['multiplier']:.1f}x")
        print(f"• Least events ({min_events}): Multiplier {min_result['multiplier']:.1f}x")
        print(f"• Event range: {max_events - min_events} events ({(max_events/min_events - 1)*100:.1f}% variation)")
        
        avg_time = sum([r.get('analysis_time', 0) for r in results if 'error' not in r]) / len([r for r in results if 'error' not in r])
        print(f"• Average analysis time: {avg_time:.1f} seconds")
    
    print("="*100)

# Quick test function for specific threshold increase
def quick_threshold_test(data, nominal, base_config, increase_factor=1.1):
    """
    Quick test for specific threshold increase
    
    Args:
        increase_factor: 1.1 = 10% increase, 1.2 = 20% increase, etc.
    """
    print(f"\n🔍 QUICK THRESHOLD TEST: {increase_factor:.1f}x increase")
    print("-" * 60)
    
    # Modified config
    test_config = base_config.copy()
    test_config['sig_event_factor'] = base_config['sig_event_factor'] * increase_factor
    test_config['stat_event_factor'] = base_config['stat_event_factor'] * increase_factor
    
    print(f"Original sig_event_factor: {base_config['sig_event_factor']:.6f}")
    print(f"New sig_event_factor:      {test_config['sig_event_factor']:.6f}")
    print(f"Original stat_event_factor: {base_config['stat_event_factor']:.6f}")
    print(f"New stat_event_factor:      {test_config['stat_event_factor']:.6f}")
    
    # Run test
    start_time = time.time()
    results = model.RBA_theta(data, nominal, test_config)
    analysis_time = time.time() - start_time
    
    sig_trad, stat_trad, sig_mcmc, stat_mcmc, tao = results
    
    total_events = len(sig_trad) + len(stat_trad) + len(sig_mcmc) + len(stat_mcmc)
    trad_events = len(sig_trad) + len(stat_trad)
    mcmc_events = len(sig_mcmc) + len(stat_mcmc)
    
    print(f"\n📊 RESULTS:")
    print(f"Total events: {total_events}")
    print(f"Traditional:  {trad_events} events")
    print(f"MCMC:         {mcmc_events} events")
    print(f"Analysis time: {analysis_time:.1f} seconds")
    
    return results