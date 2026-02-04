"""
bootstrap_experience.py - Initialize Agentic System with Prior Knowledge

This module bootstraps the experience database with:
1. Expert knowledge (your hardcoded rules as training data)
2. Synthetic realistic executions
3. Past experimental results (if available)

Run this ONCE before first use of the agentic system.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Optional
import random
from dataclasses import dataclass, asdict
from collections import defaultdict

# ============================================================================
# EMBEDDED DATA STRUCTURES (standalone - no imports needed)
# ============================================================================

@dataclass
class DataCharacteristics:
    """Statistical fingerprint of a dataset"""
    sample_size: int
    num_features: int
    resolution_minutes: float
    duration_days: float
    volatility: float
    mean_value: float
    std_value: float
    skewness: float
    kurtosis: float
    stationarity_score: float
    trend_strength: float
    seasonality_strength: float
    num_regimes: int
    regime_persistence: float
    transition_frequency: float
    entropy: float
    noise_level: float
    prediction_difficulty: float
    autocorrelation_1h: float
    autocorrelation_6h: float
    autocorrelation_24h: float
    event_frequency: float
    ramp_rate_mean: float
    ramp_rate_max: float
    lyapunov_exponent: float
    hurst_exponent: float
    sample_entropy: float
    missing_data_pct: float
    outlier_pct: float
    
    def to_dict(self):
        return asdict(self)
    
    def to_vector(self):
        return np.array([
            self.volatility, self.stationarity_score, self.trend_strength,
            self.num_regimes, self.regime_persistence, self.entropy,
            self.prediction_difficulty, self.autocorrelation_24h,
            self.event_frequency, self.lyapunov_exponent, self.hurst_exponent
        ])


@dataclass
class WorkflowExecution:
    """Record of a single workflow execution"""
    execution_id: str
    timestamp: str
    data_fingerprint: str
    data_characteristics: Dict
    dataset_name: Optional[str]
    workflow_selected: str
    selection_method: str
    confidence_score: float
    execution_time: float
    success: bool
    error_message: Optional[str]
    test_r2: Optional[float]
    test_mae: Optional[float]
    test_rmse: Optional[float]
    event_f1_score: Optional[float]
    reward: float
    
    def to_dict(self):
        return asdict(self)


class ExperienceDatabase:
    """SQLite database to store past executions"""
    
    def __init__(self, db_path: str = "agentic_experience.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                timestamp TEXT,
                data_fingerprint TEXT,
                data_characteristics TEXT,
                dataset_name TEXT,
                workflow_selected TEXT,
                selection_method TEXT,
                confidence_score REAL,
                execution_time REAL,
                success INTEGER,
                error_message TEXT,
                test_r2 REAL,
                test_mae REAL,
                test_rmse REAL,
                event_f1_score REAL,
                reward REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characteristics_summary (
                fingerprint TEXT PRIMARY KEY,
                volatility REAL,
                stationarity_score REAL,
                num_regimes INTEGER,
                prediction_difficulty REAL,
                sample_size INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✓ Experience database initialized: {self.db_path}")
    
    def store_execution(self, execution: WorkflowExecution):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.execution_id,
            execution.timestamp,
            execution.data_fingerprint,
            json.dumps(execution.data_characteristics),
            execution.dataset_name,
            execution.workflow_selected,
            execution.selection_method,
            execution.confidence_score,
            execution.execution_time,
            1 if execution.success else 0,
            execution.error_message,
            execution.test_r2,
            execution.test_mae,
            execution.test_rmse,
            execution.event_f1_score,
            execution.reward
        ))
        
        conn.commit()
        conn.close()
    
    def get_workflow_statistics(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT workflow_selected, 
                   COUNT(*) as count,
                   AVG(reward) as avg_reward,
                   MAX(reward) as max_reward,
                   AVG(test_r2) as avg_r2,
                   AVG(event_f1_score) as avg_f1
            FROM executions
            WHERE success = 1
            GROUP BY workflow_selected
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {}
        for row in rows:
            stats[row[0]] = {
                'count': row[1],
                'avg_reward': row[2] or 0.0,
                'max_reward': row[3] or 0.0,
                'avg_r2': row[4] or 0.0,
                'avg_f1': row[5] or 0.0
            }
        
        return stats


# ============================================================================
# EXPERT KNOWLEDGE BASE
# ============================================================================

class ExpertKnowledgeBase:
    """
    Encode expert knowledge (your hardcoded rules) as synthetic training data
    """
    
    def __init__(self):
        self.knowledge_rules = self._define_expert_rules()
    
    def _define_expert_rules(self) -> List[Dict]:
        """
        Define expert knowledge as rules:
        "For data with X characteristics, workflow Y performs at Z level"
        
        These are YOUR insights from experiments!
        """
        rules = [
            # ================================================================
            # RULE 1: SHORT HORIZON + HIGH VOLATILITY → LSTM (workflow3)
            # ================================================================
            {
                'name': 'short_horizon_high_volatility',
                'description': 'High volatility data with short prediction horizon',
                'characteristics': {
                    'volatility': (0.3, 0.6),  # High
                    'stationarity_score': (0.3, 0.7),  # Medium
                    'num_regimes': (2, 4),
                    'regime_persistence': (0.4, 0.7),
                    'prediction_difficulty': (0.4, 0.7),  # Medium-hard
                    'autocorrelation_24h': (0.5, 0.8),  # Medium-strong
                    'duration_days': (100, 500),
                    'sample_size': (50000, 200000)
                },
                'best_workflow': 'workflow3',  # LSTM
                'performance': {
                    'test_r2': (0.85, 0.92),
                    'test_mae': (80, 150),
                    'event_f1_score': (0.78, 0.88),
                    'execution_time': (1200, 2400)
                },
                'confidence': 0.85,
                'reason': 'LSTM handles temporal dependencies well in volatile conditions'
            },
            
            # ================================================================
            # RULE 2: LOW VOLATILITY + STATIONARY → SARIMAX (workflow1)
            # ================================================================
            {
                'name': 'low_volatility_stationary',
                'description': 'Stable, stationary data with clear patterns',
                'characteristics': {
                    'volatility': (0.05, 0.2),  # Low
                    'stationarity_score': (0.7, 1.0),  # High (stationary)
                    'num_regimes': (1, 2),  # Few regimes
                    'regime_persistence': (0.7, 1.0),  # Persistent
                    'prediction_difficulty': (0.1, 0.4),  # Easy-medium
                    'autocorrelation_24h': (0.6, 0.9),  # Strong
                    'duration_days': (200, 1000),
                    'sample_size': (80000, 300000)
                },
                'best_workflow': 'workflow1',  # SARIMAX
                'performance': {
                    'test_r2': (0.88, 0.94),
                    'test_mae': (60, 120),
                    'event_f1_score': (0.82, 0.90),
                    'execution_time': (600, 1200)
                },
                'confidence': 0.90,
                'reason': 'SARIMAX excels at stationary time series with clear patterns'
            },
            
            # ================================================================
            # RULE 3: MULTIPLE REGIMES + REGIME SHIFTS → TRANSFORMER (workflow4)
            # ================================================================
            {
                'name': 'multi_regime_shifts',
                'description': 'Complex data with frequent regime changes',
                'characteristics': {
                    'volatility': (0.25, 0.5),
                    'stationarity_score': (0.2, 0.6),  # Non-stationary
                    'num_regimes': (3, 5),  # Many regimes
                    'regime_persistence': (0.3, 0.6),  # Low persistence
                    'transition_frequency': (3.0, 8.0),  # Frequent transitions
                    'prediction_difficulty': (0.5, 0.8),  # Hard
                    'autocorrelation_24h': (0.3, 0.6),  # Weak
                    'duration_days': (200, 800),
                    'sample_size': (100000, 400000)
                },
                'best_workflow': 'workflow4',  # Transformer
                'performance': {
                    'test_r2': (0.88, 0.95),
                    'test_mae': (70, 140),
                    'event_f1_score': (0.82, 0.92),
                    'execution_time': (1800, 3600)
                },
                'confidence': 0.88,
                'reason': 'Transformer attention mechanism handles regime shifts effectively'
            },
            
            # ================================================================
            # RULE 4: HIGH AUTOCORRELATION + SEASONAL → LSTM (workflow3)
            # ================================================================
            {
                'name': 'seasonal_patterns',
                'description': 'Strong seasonal patterns with high autocorrelation',
                'characteristics': {
                    'volatility': (0.15, 0.35),
                    'stationarity_score': (0.5, 0.8),
                    'seasonality_strength': (0.6, 0.9),  # Strong seasonality
                    'autocorrelation_24h': (0.7, 0.95),  # Very strong
                    'prediction_difficulty': (0.3, 0.6),
                    'duration_days': (300, 1000),
                    'sample_size': (100000, 400000)
                },
                'best_workflow': 'workflow3',  # LSTM
                'performance': {
                    'test_r2': (0.89, 0.94),
                    'test_mae': (65, 130),
                    'event_f1_score': (0.83, 0.91),
                    'execution_time': (1400, 2600)
                },
                'confidence': 0.87,
                'reason': 'LSTM memory cells capture seasonal dependencies well'
            },
            
            # ================================================================
            # RULE 5: MODERATE EVERYTHING → LSTM (workflow3) [Default]
            # ================================================================
            {
                'name': 'balanced_moderate',
                'description': 'Moderate characteristics across the board',
                'characteristics': {
                    'volatility': (0.2, 0.4),
                    'stationarity_score': (0.4, 0.7),
                    'num_regimes': (2, 3),
                    'regime_persistence': (0.5, 0.8),
                    'prediction_difficulty': (0.3, 0.6),
                    'autocorrelation_24h': (0.5, 0.8),
                    'duration_days': (150, 600),
                    'sample_size': (60000, 250000)
                },
                'best_workflow': 'workflow3',  # LSTM (balanced choice)
                'performance': {
                    'test_r2': (0.83, 0.90),
                    'test_mae': (90, 160),
                    'event_f1_score': (0.76, 0.86),
                    'execution_time': (1300, 2500)
                },
                'confidence': 0.75,
                'reason': 'LSTM is a balanced choice for general-purpose forecasting'
            },
            
            # ================================================================
            # RULE 6: HIGH CHAOS + LOW PREDICTABILITY → TRANSFORMER (workflow4)
            # ================================================================
            {
                'name': 'chaotic_difficult',
                'description': 'Chaotic, difficult to predict data',
                'characteristics': {
                    'volatility': (0.4, 0.7),
                    'lyapunov_exponent': (0.3, 0.8),  # High chaos
                    'entropy': (0.7, 1.0),  # High entropy
                    'prediction_difficulty': (0.6, 0.9),  # Very hard
                    'autocorrelation_24h': (0.2, 0.5),  # Weak
                    'noise_level': (0.4, 0.8),  # High noise
                    'duration_days': (200, 700),
                    'sample_size': (80000, 300000)
                },
                'best_workflow': 'workflow4',  # Transformer (most powerful)
                'performance': {
                    'test_r2': (0.78, 0.88),  # Lower due to difficulty
                    'test_mae': (100, 200),
                    'event_f1_score': (0.72, 0.84),
                    'execution_time': (2000, 3800)
                },
                'confidence': 0.82,
                'reason': 'Transformer best equipped for complex, chaotic patterns'
            },
            
            # ================================================================
            # RULE 7: SMOOTH TREND + LOW NOISE → SARIMAX (workflow1)
            # ================================================================
            {
                'name': 'smooth_trend',
                'description': 'Clean data with clear trend and low noise',
                'characteristics': {
                    'volatility': (0.08, 0.18),
                    'trend_strength': (0.6, 0.9),  # Strong trend
                    'noise_level': (0.1, 0.3),  # Low noise
                    'prediction_difficulty': (0.15, 0.35),  # Easy
                    'stationarity_score': (0.6, 0.9),
                    'duration_days': (250, 900),
                    'sample_size': (100000, 350000)
                },
                'best_workflow': 'workflow1',  # SARIMAX
                'performance': {
                    'test_r2': (0.90, 0.96),  # Excellent
                    'test_mae': (50, 100),
                    'event_f1_score': (0.85, 0.93),
                    'execution_time': (700, 1300)
                },
                'confidence': 0.92,
                'reason': 'SARIMAX perfect for clean trending data'
            },
            
            # ================================================================
            # RULE 8: SPARSE DATA + UNCERTAINTY → LSTM with Transfer (workflow3)
            # ================================================================
            {
                'name': 'sparse_uncertain',
                'description': 'Limited data with high uncertainty',
                'characteristics': {
                    'sample_size': (10000, 60000),  # Small dataset
                    'duration_days': (30, 180),  # Short duration
                    'prediction_difficulty': (0.5, 0.8),
                    'entropy': (0.6, 0.9),
                    'volatility': (0.25, 0.5),
                    'missing_data_pct': (2.0, 10.0)  # Some missing data
                },
                'best_workflow': 'workflow3',  # LSTM (can use transfer learning)
                'performance': {
                    'test_r2': (0.75, 0.85),
                    'test_mae': (110, 180),
                    'event_f1_score': (0.70, 0.82),
                    'execution_time': (800, 1600)
                },
                'confidence': 0.70,
                'reason': 'LSTM with transfer learning works with limited data'
            },
            
            # ================================================================
            # RULE 9: NON-LINEAR PATTERNS + FEATURE INTERACTIONS → RF-MCMC (workflow2)
            # ================================================================
            {
                'name': 'nonlinear_feature_interactions',
                'description': 'Non-linear relationships with strong feature interactions',
                'characteristics': {
                    'volatility': (0.22, 0.45),
                    'stationarity_score': (0.4, 0.7),
                    'num_regimes': (2, 3),
                    'regime_persistence': (0.5, 0.75),
                    'prediction_difficulty': (0.4, 0.65),
                    'autocorrelation_24h': (0.4, 0.7),
                    'entropy': (0.5, 0.8),
                    'noise_level': (0.3, 0.6),
                    'duration_days': (180, 700),
                    'sample_size': (70000, 280000)
                },
                'best_workflow': 'workflow2',  # RF-MCMC-RBA
                'performance': {
                    'test_r2': (0.84, 0.91),
                    'test_mae': (75, 145),
                    'event_f1_score': (0.79, 0.88),
                    'execution_time': (1000, 2000)
                },
                'confidence': 0.83,
                'reason': 'Random Forest captures non-linear interactions; MCMC provides uncertainty quantification'
            },
            
            # ================================================================
            # RULE 10: MODERATE NOISE + MISSING DATA → RF-MCMC (workflow2)
            # ================================================================
            {
                'name': 'robust_to_noise_missing',
                'description': 'Noisy data with some missing values requiring robust methods',
                'characteristics': {
                    'volatility': (0.25, 0.5),
                    'stationarity_score': (0.35, 0.65),
                    'num_regimes': (2, 4),
                    'prediction_difficulty': (0.45, 0.7),
                    'noise_level': (0.35, 0.65),  # Moderate-high noise
                    'missing_data_pct': (3.0, 8.0),  # Some missing data
                    'autocorrelation_24h': (0.45, 0.75),
                    'entropy': (0.55, 0.85),
                    'duration_days': (150, 600),
                    'sample_size': (60000, 250000)
                },
                'best_workflow': 'workflow2',  # RF-MCMC-RBA
                'performance': {
                    'test_r2': (0.80, 0.88),
                    'test_mae': (85, 160),
                    'event_f1_score': (0.75, 0.85),
                    'execution_time': (1100, 2100)
                },
                'confidence': 0.80,
                'reason': 'Random Forest robust to noise and missing data; MCMC handles uncertainty'
            },
        ]
        
        return rules
    
    def generate_synthetic_execution(self, rule: Dict, variation_factor: float = 0.1) -> WorkflowExecution:
        """
        Generate a synthetic execution based on expert rule
        
        Args:
            rule: Expert rule definition
            variation_factor: How much to vary the characteristics (0-1)
        """
        # Generate synthetic characteristics
        characteristics = DataCharacteristics(
            sample_size=self._random_from_range(rule['characteristics'].get('sample_size', (100000, 200000))),
            num_features=random.randint(8, 15),
            resolution_minutes=random.choice([10, 15, 30, 60]),
            duration_days=self._random_from_range(rule['characteristics'].get('duration_days', (200, 600))),
            
            volatility=self._random_from_range(rule['characteristics'].get('volatility', (0.2, 0.4)), variation_factor),
            mean_value=random.uniform(1000, 5000),
            std_value=random.uniform(100, 1000),
            skewness=random.uniform(-0.5, 0.5),
            kurtosis=random.uniform(2, 4),
            
            stationarity_score=self._random_from_range(rule['characteristics'].get('stationarity_score', (0.5, 0.8)), variation_factor),
            trend_strength=self._random_from_range(rule['characteristics'].get('trend_strength', (0.2, 0.6)), variation_factor),
            seasonality_strength=self._random_from_range(rule['characteristics'].get('seasonality_strength', (0.3, 0.7)), variation_factor),
            
            num_regimes=int(self._random_from_range(rule['characteristics'].get('num_regimes', (2, 3)))),
            regime_persistence=self._random_from_range(rule['characteristics'].get('regime_persistence', (0.5, 0.8)), variation_factor),
            transition_frequency=self._random_from_range(rule['characteristics'].get('transition_frequency', (1.5, 4.0)), variation_factor),
            
            entropy=self._random_from_range(rule['characteristics'].get('entropy', (0.4, 0.7)), variation_factor),
            noise_level=self._random_from_range(rule['characteristics'].get('noise_level', (0.2, 0.5)), variation_factor),
            prediction_difficulty=self._random_from_range(rule['characteristics'].get('prediction_difficulty', (0.3, 0.6)), variation_factor),
            
            autocorrelation_1h=random.uniform(0.7, 0.95),
            autocorrelation_6h=random.uniform(0.5, 0.8),
            autocorrelation_24h=self._random_from_range(rule['characteristics'].get('autocorrelation_24h', (0.5, 0.8)), variation_factor),
            
            event_frequency=random.uniform(2.0, 6.0),
            ramp_rate_mean=random.uniform(0.1, 0.4),
            ramp_rate_max=random.uniform(0.5, 1.5),
            
            lyapunov_exponent=self._random_from_range(rule['characteristics'].get('lyapunov_exponent', (0.1, 0.4)), variation_factor),
            hurst_exponent=random.uniform(0.5, 0.8),
            sample_entropy=random.uniform(0.4, 0.8),
            
            missing_data_pct=self._random_from_range(rule['characteristics'].get('missing_data_pct', (0.0, 2.0)), variation_factor),
            outlier_pct=random.uniform(0.5, 3.0)
        )
        
        # Generate synthetic performance metrics
        perf = rule['performance']
        test_r2 = self._random_from_range(perf['test_r2'], variation_factor)
        test_mae = self._random_from_range(perf['test_mae'], variation_factor)
        test_rmse = test_mae * random.uniform(1.1, 1.3)  # RMSE typically 10-30% higher
        event_f1 = self._random_from_range(perf['event_f1_score'], variation_factor)
        exec_time = self._random_from_range(perf['execution_time'], variation_factor)
        
        # Calculate reward
        reward = self._calculate_reward(test_r2, test_mae, event_f1, exec_time)
        
        # Create execution record
        char_json = json.dumps(characteristics.to_dict(), sort_keys=True)
        fingerprint = hashlib.md5(char_json.encode()).hexdigest()
        
        # Generate realistic timestamp (past 6 months)
        days_ago = random.randint(1, 180)
        timestamp = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        
        execution_id = f"bootstrap_{rule['name']}_{random.randint(1000, 9999)}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            timestamp=timestamp,
            data_fingerprint=fingerprint,
            data_characteristics=characteristics.to_dict(),
            dataset_name=f"synthetic_{rule['name']}",
            workflow_selected=rule['best_workflow'],
            selection_method='expert_bootstrap',
            confidence_score=rule['confidence'],
            execution_time=exec_time,
            success=True,
            error_message=None,
            test_r2=test_r2,
            test_mae=test_mae,
            test_rmse=test_rmse,
            event_f1_score=event_f1,
            reward=reward
        )
        
        return execution
    
    def _random_from_range(self, range_tuple, variation: float = 0.0):
        """Generate random value from range with optional variation"""
        low, high = range_tuple
        base = random.uniform(low, high)
        
        if variation > 0:
            # Add variation
            var_amount = (high - low) * variation
            base += random.uniform(-var_amount, var_amount)
            # Clip to valid range
            base = max(low, min(high, base))
        
        return base
    
    def _calculate_reward(self, test_r2, test_mae, event_f1, execution_time):
        """Calculate reward score"""
        reward = 0.0
        
        # R² component (0-0.4)
        reward += min(test_r2, 1.0) * 0.4
        
        # MAE component (0-0.2, inverted)
        mae_norm = max(0, 1 - test_mae / 500)
        reward += mae_norm * 0.2
        
        # Event F1 component (0-0.3)
        reward += min(event_f1, 1.0) * 0.3
        
        # Efficiency component (0-0.1)
        time_norm = max(0, 1 - execution_time / 3600)
        reward += time_norm * 0.1
        
        return reward


# ============================================================================
# BOOTSTRAP ORCHESTRATOR
# ============================================================================

class DatabaseBootstrap:
    """
    Bootstrap the experience database with prior knowledge
    """
    
    def __init__(self, db_path: str = "agentic_experience.db"):
        self.db_path = db_path
        self.experience_db = ExperienceDatabase(db_path)
        self.expert_kb = ExpertKnowledgeBase()
    
    def bootstrap(self, executions_per_rule: int = 5, 
                  add_competing_workflows: bool = True):
        """
        Bootstrap database with synthetic executions
        
        Args:
            executions_per_rule: How many synthetic executions per expert rule
            add_competing_workflows: Also add lower-performing competing workflows
        """
        print("\n" + "="*80)
        print("🚀 BOOTSTRAPPING AGENTIC EXPERIENCE DATABASE")
        print("="*80)
        
        total_executions = 0
        
        for rule_idx, rule in enumerate(self.expert_kb.knowledge_rules, 1):
            print(f"\n📚 Rule {rule_idx}/{len(self.expert_kb.knowledge_rules)}: {rule['name']}")
            print(f"   {rule['description']}")
            print(f"   Best workflow: {rule['best_workflow']}")
            print(f"   Confidence: {rule['confidence']:.1%}")
            
            # Generate executions for best workflow
            for i in range(executions_per_rule):
                execution = self.expert_kb.generate_synthetic_execution(
                    rule, 
                    variation_factor=0.15  # 15% variation
                )
                self.experience_db.store_execution(execution)
                total_executions += 1
            
            print(f"   ✓ Generated {executions_per_rule} executions for {rule['best_workflow']}")
            
            # Add competing workflows (with lower performance)
            if add_competing_workflows:
                competing_count = self._add_competing_workflows(rule, executions_per_rule // 2)
                total_executions += competing_count
                print(f"   ✓ Generated {competing_count} competing workflow executions")
        
        print("\n" + "="*80)
        print(f"✅ BOOTSTRAP COMPLETE")
        print("="*80)
        print(f"   Total executions added: {total_executions}")
        print(f"   Database: {self.db_path}")
        print(f"\n💡 The agentic system now has {total_executions} past experiences to learn from!")
        
        # Show statistics
        self._show_bootstrap_statistics()
    
    def _add_competing_workflows(self, rule: Dict, count: int) -> int:
        """
        Add executions for competing workflows (with degraded performance)
        
        NOW INCLUDES ALL FOUR WORKFLOWS: workflow1, workflow2, workflow3, workflow4
        """
        workflows = ['workflow1', 'workflow2', 'workflow3', 'workflow4']
        best_workflow = rule['best_workflow']
        competing_workflows = [w for w in workflows if w != best_workflow]
        
        added = 0
        
        for workflow in competing_workflows:
            for _ in range(count):
                # Generate execution with degraded performance
                execution = self.expert_kb.generate_synthetic_execution(
                    rule,
                    variation_factor=0.2
                )
                
                # Override workflow
                execution.workflow_selected = workflow
                
                # Degrade performance (5-15% worse)
                degradation = random.uniform(0.85, 0.95)
                
                if execution.test_r2:
                    execution.test_r2 *= degradation
                if execution.event_f1_score:
                    execution.event_f1_score *= degradation
                if execution.test_mae:
                    execution.test_mae /= degradation
                
                # Recalculate reward
                execution.reward = self.expert_kb._calculate_reward(
                    execution.test_r2,
                    execution.test_mae,
                    execution.event_f1_score,
                    execution.execution_time
                )
                
                self.experience_db.store_execution(execution)
                added += 1
        
        return added
    
    def _show_bootstrap_statistics(self):
        """Display bootstrap statistics"""
        stats = self.experience_db.get_workflow_statistics()
        
        print("\n📊 BOOTSTRAP STATISTICS")
        print("-" * 80)
        
        for workflow, wf_stats in sorted(stats.items()):
            print(f"\n{workflow}:")
            print(f"   Executions: {wf_stats['count']}")
            print(f"   Avg Reward: {wf_stats['avg_reward']:.3f}")
            print(f"   Avg R²: {wf_stats['avg_r2']:.3f}")
            print(f"   Avg F1: {wf_stats['avg_f1']:.3f}")
    
    def clear_database(self):
        """Clear existing database"""
        response = input("\n⚠️  This will DELETE all existing experience data. Continue? (yes/no): ")
        if response.lower() == 'yes':
            import os
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.experience_db = ExperienceDatabase(self.db_path)
            print("✓ Database cleared")
        else:
            print("Operation cancelled")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main bootstrap function
    """
    print("\n" + "="*80)
    print("🎯 AGENTIC SYSTEM DATABASE BOOTSTRAP")
    print("="*80)
    print("\nThis will initialize the experience database with expert knowledge.")
    print("This should be run ONCE before first use of the agentic system.")
    
    print("\n" + "-"*80)
    print("Options:")
    print("-" * 40)
    print("1. Bootstrap new database (recommended)")
    print("2. Clear existing database and bootstrap")
    print("3. Add more synthetic data to existing database")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    bootstrapper = DatabaseBootstrap()
    
    if choice == '1':
        print("\n✓ Creating new database with expert knowledge...")
        bootstrapper.bootstrap(
            executions_per_rule=5,  # 5 executions per rule
            add_competing_workflows=True
        )
    
    elif choice == '2':
        bootstrapper.clear_database()
        print("\n✓ Bootstrapping fresh database...")
        bootstrapper.bootstrap(
            executions_per_rule=5,
            add_competing_workflows=True
        )
    
    elif choice == '3':
        print("\n✓ Adding more synthetic data...")
        bootstrapper.bootstrap(
            executions_per_rule=3,
            add_competing_workflows=True
        )
    
    elif choice == '4':
        print("Exiting...")
        return
    
    print("\n" + "="*80)
    print("✅ READY TO USE AGENTIC SYSTEM!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run: python main_agentic.py")
    print("2. The system will use this bootstrapped knowledge")
    print("3. As you run real predictions, it will learn and improve")
    print("\n💡 The system starts with expert knowledge and improves with real data!")


if __name__ == '__main__':
    main()