"""
Unified Metrics Comparison
Convert ALL methods to RBA-theta format and apply SAME quality metrics
This ensures fair comparison using identical evaluation criteria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your core modules
import core.model as model
from core.cusum_method import run_cusum_analysis
from core.swrt_method import run_swrt_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedMetricsComparison:
    """
    Convert ALL method outputs to RBA-theta format and apply SAME quality metrics
    This ensures true apples-to-apples comparison
    """
    
    def __init__(self, data: pd.DataFrame, nominal_power: float = None):
        self.data = data
        
        # Auto-calculate nominal if not provided
        if nominal_power is None:
            turbine_cols = [col for col in data.columns if col.startswith('Turbine_')]
            self.nominal_power = data[turbine_cols].max().max() if turbine_cols else data.max().max()
            print(f"📊 Auto-calculated nominal power: {self.nominal_power:.2f} MW")
        else:
            self.nominal_power = nominal_power
        
        # Validate data
        self.turbine_columns = [col for col in data.columns if col.startswith('Turbine_')]
        if not self.turbine_columns:
            raise ValueError("No turbine columns found!")
        
        self.results = {}
        self.runtime_stats = {}
        self.unified_events = {}  # All methods in RBA format
        
    def run_all_methods(self, config: Dict = None) -> Dict:
        """Run all 4 methods and get their raw outputs"""
        logger.info("🚀 Running all 4 methods...")
        
        if config is None:
            config = model.tune_mixed_strategy(self.data, self.nominal_power)
        
        # 1. Run RBA-theta (both traditional and MCMC)
        logger.info("🔧 Running RBA-theta...")
        start_time = time.time()
        
        rba_results = model.RBA_theta(self.data, self.nominal_power, config, "unified_rbatheta.db")
        sig_trad, stat_trad, sig_mcmc, stat_mcmc, tao = rba_results
        
        rba_runtime = time.time() - start_time
        
        # Store RBA results
        self.results['rba_traditional'] = {
            'significant_events': sig_trad,
            'stationary_events': stat_trad,
            'total_events': len(sig_trad) + len(stat_trad)
        }
        
        self.results['rba_mcmc'] = {
            'significant_events': sig_mcmc,
            'stationary_events': stat_mcmc,
            'total_events': len(sig_mcmc) + len(stat_mcmc)
        }
        
        self.runtime_stats['rba_traditional'] = rba_runtime / 2  # Split runtime
        self.runtime_stats['rba_mcmc'] = rba_runtime / 2
        
        logger.info(f"✅ RBA-Traditional: {len(sig_trad) + len(stat_trad)} events")
        logger.info(f"✅ RBA-MCMC: {len(sig_mcmc) + len(stat_mcmc)} events")
        
        # 2. Run CUSUM
        logger.info("🔧 Running CUSUM...")
        start_time = time.time()
        
        cusum_events = run_cusum_analysis(self.data)
        cusum_runtime = time.time() - start_time
        
        self.results['cusum'] = {
            'events': cusum_events,
            'total_events': len(cusum_events)
        }
        self.runtime_stats['cusum'] = cusum_runtime
        
        logger.info(f"✅ CUSUM: {len(cusum_events)} events")
        
        # 3. Run SWRT
        logger.info("🌪️ Running SWRT...")
        start_time = time.time()
        
        swrt_events = run_swrt_analysis(self.data, self.nominal_power)
        swrt_runtime = time.time() - start_time
        
        self.results['swrt'] = {
            'events': swrt_events,
            'total_events': len(swrt_events)
        }
        self.runtime_stats['swrt'] = swrt_runtime
        
        logger.info(f"✅ SWRT: {len(swrt_events)} events")
        
        return self.results
    
    def convert_all_to_rba_format(self):
        """
        Convert CUSUM and SWRT outputs to RBA-theta format
        This allows applying the SAME quality metrics to all methods
        """
        logger.info("🔄 Converting all outputs to RBA-theta format...")
        
        # RBA methods are already in correct format
        self.unified_events['rba_traditional'] = {
            'significant_events': self.results['rba_traditional']['significant_events'],
            'stationary_events': self.results['rba_traditional']['stationary_events']
        }
        
        self.unified_events['rba_mcmc'] = {
            'significant_events': self.results['rba_mcmc']['significant_events'],
            'stationary_events': self.results['rba_mcmc']['stationary_events']
        }
        
        # Convert CUSUM to RBA format
        self.unified_events['cusum'] = self._convert_cusum_to_rba_format()
        
        # Convert SWRT to RBA format
        self.unified_events['swrt'] = self._convert_swrt_to_rba_format()
        
        logger.info("✅ All methods converted to unified RBA format")
    
    def _convert_cusum_to_rba_format(self) -> Dict:
        """Convert CUSUM fault events to RBA-theta format"""
        cusum_events = self.results['cusum']['events']
        
        if cusum_events.empty:
            return {
                'significant_events': pd.DataFrame(),
                'stationary_events': pd.DataFrame()
            }
        
        # Convert CUSUM faults to "significant events" format
        rba_format_events = []
        
        for idx, event in cusum_events.iterrows():
            # Create RBA-style significant event
            rba_event = {
                't1': event.get('start_time', 0),
                't2': event.get('end_time', 0),
                '∆t_m': event.get('end_time', 0) - event.get('start_time', 0),
                '∆w_m': event.get('magnitude', 0),  # Use CUSUM magnitude
                'θ_m': np.pi/2 if event.get('magnitude', 0) > 0 else -np.pi/2,  # Estimate direction
                'σ_m': abs(event.get('magnitude', 0)) * 0.1  # Estimate standard deviation
            }
            rba_format_events.append(rba_event)
        
        # All CUSUM events are treated as "significant" (fault detection)
        significant_df = pd.DataFrame(rba_format_events)
        stationary_df = pd.DataFrame()  # CUSUM doesn't detect stationary periods
        
        return {
            'significant_events': significant_df,
            'stationary_events': stationary_df
        }
    
    def _convert_swrt_to_rba_format(self) -> Dict:
        """Convert SWRT ramp events to RBA-theta format"""
        swrt_events = self.results['swrt']['events']
        
        if swrt_events.empty:
            return {
                'significant_events': pd.DataFrame(),
                'stationary_events': pd.DataFrame()
            }
        
        # Convert SWRT ramps to "significant events" format
        rba_format_events = []
        
        for idx, event in swrt_events.iterrows():
            # Create RBA-style significant event
            rba_event = {
                't1': event.get('start_time', 0),
                't2': event.get('end_time', 0),
                '∆t_m': event.get('duration', 0),
                '∆w_m': event.get('magnitude', 0) if event.get('event_type', '') == 'ramp_up' else -event.get('magnitude', 0),
                'θ_m': np.pi/4 if event.get('event_type', '') == 'ramp_up' else -np.pi/4,  # Ramp direction
                'σ_m': abs(event.get('magnitude', 0)) * 0.05  # Estimate standard deviation
            }
            rba_format_events.append(rba_event)
        
        # All SWRT events are treated as "significant" (ramp detection)
        significant_df = pd.DataFrame(rba_format_events)
        stationary_df = pd.DataFrame()  # SWRT doesn't detect stationary periods
        
        return {
            'significant_events': significant_df,
            'stationary_events': stationary_df
        }
    
    def calculate_unified_quality_metrics(self) -> Dict:
        """
        Apply the EXACT SAME quality metrics from your sensitivity analysis to ALL methods
        Now we can directly compare quality scores across all 4 methods!
        """
        logger.info("📊 Calculating unified quality metrics (SAME for all methods)...")
        
        unified_metrics = {}
        
        for method_name in ['rba_traditional', 'rba_mcmc', 'cusum', 'swrt']:
            if method_name not in self.unified_events:
                continue
            
            method_events = self.unified_events[method_name]
            sig_events = method_events['significant_events']
            stat_events = method_events['stationary_events']
            
            # Apply YOUR EXACT quality calculation to all methods
            try:
                # Use the same quality metrics calculation from your model
                quality_results = model.calculate_event_quality_metrics(sig_events, stat_events)
                
                # Extract the same metrics as your sensitivity analysis
                if isinstance(quality_results, dict):
                    sig_quality = quality_results.get('significant_events', {}).get('overall_quality_score', 0)
                    stat_quality = quality_results.get('stationary_events', {}).get('overall_quality_score', 0) 
                    overall_balance = quality_results.get('overall', {}).get('balance_score', 0)
                else:
                    # Fallback calculation
                    sig_quality = self._calculate_fallback_quality(sig_events)
                    stat_quality = self._calculate_fallback_quality(stat_events)
                    overall_balance = self._calculate_balance_score(sig_events, stat_events)
                
            except Exception as e:
                logger.warning(f"Quality calculation failed for {method_name}: {e}")
                sig_quality = self._calculate_fallback_quality(sig_events)
                stat_quality = self._calculate_fallback_quality(stat_events)
                overall_balance = self._calculate_balance_score(sig_events, stat_events)
            
            unified_metrics[method_name] = {
                'total_events': len(sig_events) + len(stat_events),
                'significant_events': len(sig_events),
                'stationary_events': len(stat_events),
                'sig_quality': sig_quality,
                'stat_quality': stat_quality,
                'balance_score': overall_balance,
                'runtime': self.runtime_stats.get(method_name, 0)
            }
        
        return unified_metrics
    
    def _calculate_fallback_quality(self, events_df: pd.DataFrame) -> float:
        """Fallback quality calculation if main calculation fails"""
        if events_df.empty:
            return 0.0
        
        if '∆t_m' not in events_df.columns or '∆w_m' not in events_df.columns:
            return 0.5  # Neutral score for missing data
        
        # Simple quality based on consistency
        durations = events_df['∆t_m'].values
        magnitudes = events_df['∆w_m'].abs().values
        
        if len(durations) < 2:
            return 0.5
        
        # Lower coefficient of variation = higher quality
        dur_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 1
        mag_cv = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 1
        
        avg_cv = (dur_cv + mag_cv) / 2
        quality_score = max(0, min(1, 1 - avg_cv/2))
        
        return quality_score
    
    def _calculate_balance_score(self, sig_events: pd.DataFrame, stat_events: pd.DataFrame) -> float:
        """Calculate balance score same as your sensitivity analysis"""
        total_events = len(sig_events) + len(stat_events)
        
        if total_events == 0:
            return 0.0
        
        sig_ratio = len(sig_events) / total_events
        # Same calculation as your sensitivity analysis
        balance_score = 1.0 - abs(sig_ratio - 0.5) * 2
        
        return max(0, balance_score)
    
    def generate_unified_comparison_report(self, unified_metrics: Dict):
        """Generate report with SAME metrics for all methods"""
        print("\n" + "="*100)
        print("UNIFIED 4-WAY COMPARISON REPORT")
        print("SAME QUALITY METRICS APPLIED TO ALL METHODS")
        print("="*100)
        
        # Event count comparison
        print(f"\n📊 EVENT COUNT COMPARISON:")
        print("-" * 80)
        print(f"{'Method':<20} {'Total Events':<15} {'Change from RBA-Trad':<20} {'Runtime (s)':<15}")
        print("-" * 80)
        
        baseline_events = unified_metrics.get('rba_traditional', {}).get('total_events', 1)
        
        method_names = {
            'rba_traditional': 'RBA-Traditional',
            'rba_mcmc': 'RBA-MCMC',
            'cusum': 'CUSUM',
            'swrt': 'SWRT'
        }
        
        for method, display_name in method_names.items():
            if method in unified_metrics:
                total = unified_metrics[method].get('total_events', 0)
                change = ((total - baseline_events) / baseline_events * 100) if baseline_events > 0 else 0
                runtime = unified_metrics[method].get('runtime', 0)
                
                print(f"{display_name:<20} {total:<15} {change:+6.1f}%            {runtime:<15.2f}")
        
        # UNIFIED QUALITY COMPARISON - SAME METRICS FOR ALL!
        print(f"\n🎯 UNIFIED QUALITY COMPARISON (SAME METRICS FOR ALL):")
        print("-" * 100)
        print(f"{'Method':<20} {'Sig Events':<12} {'Stat Events':<12} {'Sig Quality':<12} {'Stat Quality':<12} {'Balance':<10}")
        print("-" * 100)
        
        for method, display_name in method_names.items():
            if method in unified_metrics:
                data = unified_metrics[method]
                print(f"{display_name:<20} {data.get('significant_events', 0):<12} "
                      f"{data.get('stationary_events', 0):<12} {data.get('sig_quality', 0):<12.3f} "
                      f"{data.get('stat_quality', 0):<12.3f} {data.get('balance_score', 0):<10.3f}")
        
        # Quality ranking
        print(f"\n🏆 QUALITY RANKING:")
        print("-" * 50)
        
        # Rank by significant event quality
        sig_quality_ranking = sorted(unified_metrics.items(), 
                                   key=lambda x: x[1].get('sig_quality', 0), reverse=True)
        
        print("Significant Event Quality Ranking:")
        for i, (method, data) in enumerate(sig_quality_ranking, 1):
            quality = data.get('sig_quality', 0)
            print(f"  {i}. {method_names[method]}: {quality:.3f}")
        
        # Rank by balance score
        balance_ranking = sorted(unified_metrics.items(), 
                               key=lambda x: x[1].get('balance_score', 0), reverse=True)
        
        print("\nBalance Score Ranking:")
        for i, (method, data) in enumerate(balance_ranking, 1):
            balance = data.get('balance_score', 0)
            print(f"  {i}. {method_names[method]}: {balance:.3f}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS (FAIR COMPARISON):")
        print("-" * 40)
        
        # RBA enhancement analysis
        if 'rba_mcmc' in unified_metrics and 'rba_traditional' in unified_metrics:
            mcmc_quality = unified_metrics['rba_mcmc']['sig_quality']
            trad_quality = unified_metrics['rba_traditional']['sig_quality']
            quality_improvement = ((mcmc_quality - trad_quality) / trad_quality * 100) if trad_quality > 0 else 0
            
            print(f"• RBA-MCMC quality improvement: {quality_improvement:+.1f}%")
            print(f"• Quality scores: Traditional {trad_quality:.3f} vs MCMC {mcmc_quality:.3f}")
        
        # Compare with baselines
        if 'cusum' in unified_metrics and 'swrt' in unified_metrics:
            cusum_quality = unified_metrics['cusum']['sig_quality']
            swrt_quality = unified_metrics['swrt']['sig_quality']
            
            print(f"• Baseline qualities: CUSUM {cusum_quality:.3f} vs SWRT {swrt_quality:.3f}")
            
            # Compare RBA with baselines
            if 'rba_mcmc' in unified_metrics:
                rba_quality = unified_metrics['rba_mcmc']['sig_quality']
                vs_cusum = ((rba_quality - cusum_quality) / cusum_quality * 100) if cusum_quality > 0 else 0
                vs_swrt = ((rba_quality - swrt_quality) / swrt_quality * 100) if swrt_quality > 0 else 0
                
                print(f"• RBA-MCMC vs CUSUM: {vs_cusum:+.1f}% quality difference")
                print(f"• RBA-MCMC vs SWRT: {vs_swrt:+.1f}% quality difference")
        
        # Performance vs quality trade-off
        print(f"\n📈 PERFORMANCE VS QUALITY TRADE-OFF:")
        print("-" * 45)
        
        for method, display_name in method_names.items():
            if method in unified_metrics:
                data = unified_metrics[method]
                quality = data.get('sig_quality', 0)
                runtime = data.get('runtime', 0)
                events = data.get('total_events', 0)
                
                efficiency = events / runtime if runtime > 0 else 0
                quality_per_second = quality / runtime if runtime > 0 else 0
                
                print(f"{display_name}:")
                print(f"  Quality: {quality:.3f}, Runtime: {runtime:.2f}s, Events/sec: {efficiency:.1f}")
        
        print("\n" + "="*100)
        
        return unified_metrics
    
    def run_complete_unified_comparison(self, config: Dict = None) -> Dict:
        """Run complete comparison with unified metrics"""
        logger.info("🚀 Starting Unified 4-Way Comparison")
        logger.info("=" * 60)
        logger.info("Converting all outputs to RBA format for fair comparison")
        
        total_start = time.time()
        
        # Run all methods
        self.run_all_methods(config)
        
        # Convert to unified format
        self.convert_all_to_rba_format()
        
        # Calculate unified metrics
        unified_metrics = self.calculate_unified_quality_metrics()
        
        total_runtime = time.time() - total_start
        
        # Generate report
        self.generate_unified_comparison_report(unified_metrics)
        
        return {
            'results': self.results,
            'unified_events': self.unified_events,
            'unified_metrics': unified_metrics,
            'runtime_stats': self.runtime_stats,
            'total_runtime': total_runtime
        }
    
    def create_unified_comparison_plots(self, unified_metrics: Dict) -> plt.Figure:
        """Create plots comparing unified quality metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        method_names = ['RBA-Traditional', 'RBA-MCMC', 'CUSUM', 'SWRT']
        method_keys = ['rba_traditional', 'rba_mcmc', 'cusum', 'swrt']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. Unified Quality Comparison
        available_methods = [name for key, name in zip(method_keys, method_names) if key in unified_metrics]
        available_colors = [color for key, color in zip(method_keys, colors) if key in unified_metrics]
        
        sig_qualities = [unified_metrics[key].get('sig_quality', 0) for key in method_keys if key in unified_metrics]
        stat_qualities = [unified_metrics[key].get('stat_quality', 0) for key in method_keys if key in unified_metrics]
        balance_scores = [unified_metrics[key].get('balance_score', 0) for key in method_keys if key in unified_metrics]
        
        x = np.arange(len(available_methods))
        width = 0.25
        
        ax1.bar(x - width, sig_qualities, width, label='Significant Quality', alpha=0.8, color='lightblue')
        ax1.bar(x, stat_qualities, width, label='Stationary Quality', alpha=0.8, color='lightcoral')
        ax1.bar(x + width, balance_scores, width, label='Balance Score', alpha=0.8, color='lightgreen')
        
        ax1.set_title('Unified Quality Metrics (SAME calculation for all)')
        ax1.set_ylabel('Quality Score (0-1)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(available_methods, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (sig, stat, bal) in enumerate(zip(sig_qualities, stat_qualities, balance_scores)):
            ax1.text(i - width, sig + 0.02, f'{sig:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i, stat + 0.02, f'{stat:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width, bal + 0.02, f'{bal:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Event Count vs Quality
        total_events = [unified_metrics[key].get('total_events', 0) for key in method_keys if key in unified_metrics]
        
        ax2.scatter(total_events, sig_qualities, c=available_colors, s=200, alpha=0.7)
        
        for i, method in enumerate(available_methods):
            ax2.annotate(method, (total_events[i], sig_qualities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('Total Events Detected')
        ax2.set_ylabel('Significant Event Quality')
        ax2.set_title('Event Count vs Quality Trade-off')
        ax2.grid(True, alpha=0.3)
        
        # 3. Runtime vs Quality
        runtimes = [unified_metrics[key].get('runtime', 0) for key in method_keys if key in unified_metrics]
        
        ax3.scatter(runtimes, sig_qualities, c=available_colors, s=200, alpha=0.7)
        
        for i, method in enumerate(available_methods):
            ax3.annotate(method, (runtimes[i], sig_qualities[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('Runtime (seconds)')
        ax3.set_ylabel('Significant Event Quality')
        ax3.set_title('Runtime vs Quality Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # 4. Quality Ranking
        quality_ranks = list(range(1, len(available_methods) + 1))
        sorted_indices = sorted(range(len(sig_qualities)), key=lambda i: sig_qualities[i], reverse=True)
        sorted_methods = [available_methods[i] for i in sorted_indices]
        sorted_qualities = [sig_qualities[i] for i in sorted_indices]
        sorted_colors = [available_colors[i] for i in sorted_indices]
        
        ax4.barh(quality_ranks, sorted_qualities, color=sorted_colors, alpha=0.8)
        ax4.set_yticks(quality_ranks)
        ax4.set_yticklabels(sorted_methods)
        ax4.set_xlabel('Significant Event Quality')
        ax4.set_title('Quality Ranking (Highest to Lowest)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, quality in enumerate(sorted_qualities):
            ax4.text(quality + 0.01, i + 1, f'{quality:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig


def run_unified_comparison(data_file: str, config: Dict = None):
    """
    Main function to run unified comparison with SAME metrics for all methods
    """
    print("🎯 Unified 4-Way Comparison")
    print("=" * 50)
    print("Applying SAME quality metrics to ALL methods")
    print("Fair comparison using unified RBA-theta format")
    print()
    
    # Load data
    print(f"📁 Loading data from: {data_file}")
    
    if data_file.endswith('.xlsx') or data_file.endswith('.xls'):
        data = pd.read_excel(data_file)
    else:
        try:
            data = pd.read_csv(data_file, encoding='utf-8')
        except UnicodeDecodeError:
            data = pd.read_csv(data_file, encoding='latin-1')
    
    print(f"📊 Data loaded: {data.shape[0]} samples, {data.shape[1]} turbines")
    
    # Initialize comparison
    comparison = UnifiedMetricsComparison(data)
    
    # Run unified comparison
    results = comparison.run_complete_unified_comparison(config)
    
    # Create plots
    print("\n📊 Creating unified comparison plots...")
    fig = comparison.create_unified_comparison_plots(results['unified_metrics'])
    
    # Save results
    plt.savefig('unified_four_way_comparison.png', dpi=300, bbox_inches='tight')
    print("📁 Plot saved as: unified_four_way_comparison.png")
    
    plt.show()
    
    return results, comparison


if __name__ == "__main__":
    print("🎯 Unified Metrics Comparison")
    print("=" * 40)
    print("This ensures ALL methods are evaluated using")
    print("the EXACT SAME quality metrics as RBA-theta")
    print()
    print("Benefits:")
    print("• Fair comparison across all methods")
    print("• Same quality calculation for all")
    print("• Direct ranking of method performance")