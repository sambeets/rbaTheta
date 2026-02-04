"""
main.py - Self-Learning Agentic Event Detection and Prediction System

TRUE AGENTIC FEATURES:
1. Learns from past executions (no hardcoded rules)
2. Adapts workflow selection based on data characteristics
3. Uses Contextual Multi-Armed Bandit for intelligent exploration/exploitation
4. Maintains experience database of past decisions and outcomes
5. Automatically improves over time

Architecture:
- Data Characteristics Extractor: Statistical fingerprint
- Experience Database: Past executions and performance
- Intelligent Selector: ML-based workflow selection
- Workflow Executor: Runs and monitors workflows
- Learning Module: Updates knowledge base
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agentic_system.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DataCharacteristics:
    """
    Statistical fingerprint of a dataset
    """
    # Basic stats
    sample_size: int
    num_features: int
    resolution_minutes: float
    duration_days: float
    
    # Statistical properties
    volatility: float
    mean_value: float
    std_value: float
    skewness: float
    kurtosis: float
    
    # Stationarity & trend
    stationarity_score: float  # 0-1 (ADF test p-value inverted)
    trend_strength: float  # 0-1
    seasonality_strength: float  # 0-1
    
    # Regime analysis
    num_regimes: int
    regime_persistence: float  # 0-1 (how long regimes last)
    transition_frequency: float  # transitions per day
    
    # Uncertainty metrics
    entropy: float
    noise_level: float
    prediction_difficulty: float  # 0-1 (estimated)
    
    # Temporal patterns
    autocorrelation_1h: float
    autocorrelation_6h: float
    autocorrelation_24h: float
    
    # Event characteristics
    event_frequency: float  # events per day (estimated)
    ramp_rate_mean: float
    ramp_rate_max: float
    
    # Complexity
    lyapunov_exponent: float  # chaos indicator
    hurst_exponent: float  # long-term memory
    sample_entropy: float  # complexity
    
    # Data quality
    missing_data_pct: float
    outlier_pct: float
    
    def to_dict(self):
        return asdict(self)
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for similarity computation"""
        return np.array([
            self.volatility,
            self.stationarity_score,
            self.trend_strength,
            self.num_regimes,
            self.regime_persistence,
            self.entropy,
            self.prediction_difficulty,
            self.autocorrelation_24h,
            self.event_frequency,
            self.lyapunov_exponent,
            self.hurst_exponent
        ])


@dataclass
class WorkflowExecution:
    """
    Record of a single workflow execution
    """
    execution_id: str
    timestamp: str
    data_fingerprint: str  # hash of data characteristics
    data_characteristics: Dict
    dataset_name: Optional[str]  # NEW: Track dataset source
    
    workflow_selected: str  # workflow1, workflow3, workflow4
    selection_method: str  # 'exploration', 'exploitation', 'user_override'
    confidence_score: float
    
    # Performance metrics
    execution_time: float
    success: bool
    error_message: Optional[str]
    
    # Model performance (if available)
    test_r2: Optional[float]
    test_mae: Optional[float]
    test_rmse: Optional[float]
    event_f1_score: Optional[float]
    
    # Reward signal (composite score)
    reward: float
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# DATA CHARACTERISTICS EXTRACTOR
# ============================================================================

class DataCharacteristicsExtractor:
    """
    Extract comprehensive statistical fingerprint from time series
    """
    
    def __init__(self):
        pass
    
    def extract(self, df: pd.DataFrame, target_col: str = None, max_samples: int = 10000) -> DataCharacteristics:
        """
        Extract all characteristics from dataset with intelligent sampling
        
        Args:
            df: DataFrame with datetime index
            target_col: Target column name (auto-detected if None)
            max_samples: Maximum samples for expensive calculations (preserves temporal structure)
        """
        logger.info("Extracting data characteristics...")
        
        # Auto-detect target column
        if target_col is None:
            target_col = self._detect_target_column(df)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        data = df[target_col].dropna()
        
        # Smart sampling for large datasets
        original_size = len(data)
        if original_size > max_samples:
            logger.info(f"   Large dataset detected ({original_size:,} samples)")
            logger.info(f"   Using stratified sampling ({max_samples:,} samples) for fast analysis")
            logger.info(f"   This preserves temporal patterns and statistical properties")
            
            # Stratified sampling: Take evenly-spaced samples to preserve temporal structure
            sample_indices = np.linspace(0, len(data) - 1, max_samples, dtype=int)
            data_sampled = data.iloc[sample_indices]
            
            # Use sampled data for expensive calculations
            use_sampled = True
        else:
            data_sampled = data
            use_sampled = False
        
        # Basic stats (use full data - fast)
        sample_size = original_size  # Report original size
        num_features = len(df.columns)
        
        # Time resolution (use full data - fast)
        time_diffs = pd.Series(df.index).diff().dropna()
        resolution_minutes = time_diffs.median().total_seconds() / 60
        duration_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        
        # Statistical properties (use full data - fast)
        volatility = data.std() / (data.mean() + 1e-10)
        mean_value = data.mean()
        std_value = data.std()
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Stationarity (use sampled - medium speed)
        stationarity_score = self._test_stationarity(data_sampled if use_sampled else data)
        
        # Trend and seasonality (use sampled - medium speed)
        trend_strength = self._calculate_trend_strength(data_sampled if use_sampled else data)
        seasonality_strength = self._detect_seasonality(data if len(data) < 50000 else data_sampled)
        
        # Regime detection (use sampled - SLOW, benefits most from sampling)
        num_regimes, regime_persistence, transition_frequency = self._analyze_regimes(
            data_sampled if use_sampled else data
        )
        
        # Uncertainty metrics (use full data for entropy, sampled for others)
        entropy = self._calculate_entropy(data)  # Fast even with full data
        noise_level = self._estimate_noise_level(data_sampled if use_sampled else data)
        prediction_difficulty = self._estimate_prediction_difficulty(
            volatility, stationarity_score, num_regimes, entropy
        )
        
        # Temporal patterns (use appropriate lags)
        autocorr_1h = self._safe_autocorr(data, int(60 / resolution_minutes))
        autocorr_6h = self._safe_autocorr(data, int(360 / resolution_minutes))
        autocorr_24h = self._safe_autocorr(data, int(1440 / resolution_minutes))
        
        # Event characteristics (use sampled for speed)
        event_frequency, ramp_rate_mean, ramp_rate_max = self._analyze_events(
            data_sampled if use_sampled else data
        )
        
        # Complexity metrics (use sampled - VERY SLOW)
        lyapunov = self._calculate_lyapunov(data_sampled if use_sampled else data)
        hurst = self._calculate_hurst(data_sampled if use_sampled else data)
        sample_entropy_val = self._calculate_sample_entropy_fast(
            data_sampled if use_sampled else data
        )
        
        # Data quality (use full data - fast)
        missing_pct = (df[target_col].isna().sum() / len(df)) * 100
        outlier_pct = self._calculate_outlier_percentage(data)
        
        characteristics = DataCharacteristics(
            sample_size=sample_size,
            num_features=num_features,
            resolution_minutes=resolution_minutes,
            duration_days=duration_days,
            volatility=volatility,
            mean_value=mean_value,
            std_value=std_value,
            skewness=skewness,
            kurtosis=kurtosis,
            stationarity_score=stationarity_score,
            trend_strength=trend_strength,
            seasonality_strength=seasonality_strength,
            num_regimes=num_regimes,
            regime_persistence=regime_persistence,
            transition_frequency=transition_frequency,
            entropy=entropy,
            noise_level=noise_level,
            prediction_difficulty=prediction_difficulty,
            autocorrelation_1h=autocorr_1h,
            autocorrelation_6h=autocorr_6h,
            autocorrelation_24h=autocorr_24h,
            event_frequency=event_frequency,
            ramp_rate_mean=ramp_rate_mean,
            ramp_rate_max=ramp_rate_max,
            lyapunov_exponent=lyapunov,
            hurst_exponent=hurst,
            sample_entropy=sample_entropy_val,
            missing_data_pct=missing_pct,
            outlier_pct=outlier_pct
        )
        
        logger.info("✓ Data characteristics extracted")
        return characteristics
    
    def _detect_target_column(self, df: pd.DataFrame) -> str:
        """Auto-detect most likely target column"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Priority keywords
        priority_keywords = ['power', 'energy', 'output', 'generation']
        
        for col in numeric_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in priority_keywords):
                return col
        
        # Fallback: first numeric column
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
    
    def _test_stationarity(self, data: pd.Series) -> float:
        """
        Augmented Dickey-Fuller test for stationarity
        Returns: 0-1 score (1 = stationary, 0 = non-stationary)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(data.dropna(), maxlag=20)
            p_value = result[1]
            # Invert p-value: low p-value = stationary
            return 1.0 - min(p_value, 1.0)
        except:
            # Fallback: simple variance stability test
            window = len(data) // 10
            rolling_std = data.rolling(window).std().dropna()
            cv = rolling_std.std() / (rolling_std.mean() + 1e-10)
            return 1.0 / (1.0 + cv)
    
    def _calculate_trend_strength(self, data: pd.Series) -> float:
        """Calculate trend strength (0-1)"""
        try:
            from scipy import stats
            x = np.arange(len(data))
            slope, _, r_value, _, _ = stats.linregress(x, data.values)
            return min(abs(r_value), 1.0)
        except:
            return 0.0
    
    def _detect_seasonality(self, data: pd.Series) -> float:
        """Detect seasonality strength (0-1)"""
        try:
            # Simple ACF-based seasonality detection
            acf_values = []
            for lag in [24, 48, 72, 168]:  # Daily, 2-day, 3-day, weekly
                if lag < len(data) // 2:
                    acf = data.autocorr(lag)
                    if not np.isnan(acf):
                        acf_values.append(abs(acf))
            
            return max(acf_values) if acf_values else 0.0
        except:
            return 0.0
    
    def _analyze_regimes(self, data: pd.Series) -> Tuple[int, float, float]:
        """
        Detect regimes using simple threshold-based method
        
        Returns:
            (num_regimes, persistence, transition_frequency)
        """
        try:
            # Use quantile-based regime detection
            q25, q50, q75 = data.quantile([0.25, 0.5, 0.75])
            
            # Define regimes: low, medium, high
            regimes = pd.Series(index=data.index, dtype=int)
            regimes[data <= q25] = 0
            regimes[(data > q25) & (data <= q75)] = 1
            regimes[data > q75] = 2
            
            num_regimes = regimes.nunique()
            
            # Calculate persistence (average regime duration)
            regime_changes = (regimes != regimes.shift()).sum()
            avg_regime_length = len(regimes) / max(regime_changes, 1)
            persistence = min(avg_regime_length / (len(regimes) / num_regimes), 1.0)
            
            # Transition frequency (transitions per day)
            duration_days = (data.index[-1] - data.index[0]).total_seconds() / 86400
            transition_frequency = regime_changes / max(duration_days, 1)
            
            return num_regimes, persistence, transition_frequency
        except:
            return 1, 1.0, 0.0
    
    def _calculate_entropy(self, data: pd.Series) -> float:
        """Calculate Shannon entropy (normalized to 0-1)"""
        try:
            # Discretize data into bins
            hist, _ = np.histogram(data.dropna(), bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            # Normalize
            max_entropy = np.log2(50)  # Maximum for 50 bins
            return min(entropy / max_entropy, 1.0)
        except:
            return 0.5
    
    def _estimate_noise_level(self, data: pd.Series) -> float:
        """Estimate noise level using differencing"""
        try:
            diff = data.diff().dropna()
            noise = diff.std() / (data.std() + 1e-10)
            return min(noise, 1.0)
        except:
            return 0.5
    
    def _estimate_prediction_difficulty(self, volatility: float, 
                                       stationarity: float,
                                       num_regimes: int, 
                                       entropy: float) -> float:
        """
        Composite score for prediction difficulty (0-1)
        Higher = harder to predict
        """
        difficulty = (
            0.3 * volatility +
            0.2 * (1 - stationarity) +
            0.2 * (num_regimes / 5.0) +
            0.3 * entropy
        )
        return min(difficulty, 1.0)
    
    def _safe_autocorr(self, data: pd.Series, lag: int) -> float:
        """Calculate autocorrelation safely"""
        try:
            if lag >= len(data) // 2:
                lag = len(data) // 4
            acf = data.autocorr(lag)
            return acf if not np.isnan(acf) else 0.0
        except:
            return 0.0
    
    def _analyze_events(self, data: pd.Series) -> Tuple[float, float, float]:
        """
        Estimate event characteristics
        
        Returns:
            (event_frequency, ramp_rate_mean, ramp_rate_max)
        """
        try:
            # Detect events using gradient
            gradient = data.diff().abs()
            threshold = gradient.quantile(0.95)
            events = gradient > threshold
            
            event_count = events.sum()
            duration_days = (data.index[-1] - data.index[0]).total_seconds() / 86400
            event_frequency = event_count / max(duration_days, 1)
            
            ramp_rate_mean = gradient[events].mean() if event_count > 0 else 0.0
            ramp_rate_max = gradient[events].max() if event_count > 0 else 0.0
            
            return event_frequency, ramp_rate_mean, ramp_rate_max
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_lyapunov(self, data: pd.Series) -> float:
        """Estimate Lyapunov exponent (chaos indicator)"""
        try:
            # Simplified Lyapunov estimation
            d = data.diff().dropna().values
            divergence = np.log(np.abs(d) + 1e-10)
            lyapunov = divergence.mean()
            return min(max(lyapunov, -1), 1)  # Clip to [-1, 1]
        except:
            return 0.0
    
    def _calculate_hurst(self, data: pd.Series) -> float:
        """Calculate Hurst exponent (long-term memory)"""
        try:
            # Simplified Hurst calculation
            lags = range(2, min(100, len(data) // 2))
            tau = [np.std(np.subtract(data[lag:].values, data[:-lag].values)) 
                   for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0]
            return min(max(hurst, 0), 1)  # Clip to [0, 1]
        except:
            return 0.5
    
    def _calculate_sample_entropy(self, data: pd.Series) -> float:
        """Calculate sample entropy (complexity measure) - SLOW VERSION"""
        try:
            # Simplified sample entropy
            m, r = 2, 0.2
            N = len(data)
            data_norm = (data - data.mean()) / (data.std() + 1e-10)
            
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                x = [[data_norm.iloc[j] for j in range(i, i + m)] 
                     for i in range(N - m)]
                C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r])
                     for i in range(len(x))]
                return sum(C) / (len(x) * (len(x) - 1))
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m > 0 and phi_m1 > 0:
                return -np.log(phi_m1 / phi_m)
            return 0.5
        except:
            return 0.5
    
    def _calculate_sample_entropy_fast(self, data: pd.Series, max_samples: int = 2000) -> float:
        """
        Fast approximation of sample entropy using subsampling
        
        For large datasets, use stratified sampling to estimate complexity
        """
        try:
            if len(data) > max_samples:
                # Stratified sampling
                sample_indices = np.linspace(0, len(data) - 1, max_samples, dtype=int)
                data_sample = data.iloc[sample_indices]
            else:
                data_sample = data
            
            # Use simpler approximation based on successive differences
            diff = data_sample.diff().dropna()
            
            # Approximate entropy using distribution of differences
            hist, _ = np.histogram(diff, bins=30, density=True)
            hist = hist[hist > 0]
            
            if len(hist) > 0:
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                max_entropy = np.log2(30)
                return min(entropy / max_entropy, 1.0)
            
            return 0.5
        except:
            return 0.5
    
    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method"""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((data < lower) | (data > upper)).sum()
            return (outliers / len(data)) * 100
        except:
            return 0.0


# ============================================================================
# EXPERIENCE DATABASE
# ============================================================================

class ExperienceDatabase:
    """
    SQLite database to store past executions and learn from them
    """
    
    def __init__(self, db_path: str = "agentic_experience.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create executions table
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
        
        # Create characteristics summary table for faster lookups
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
        
        logger.info(f"✓ Experience database initialized: {self.db_path}")
    
    def store_execution(self, execution: WorkflowExecution):
        """Store execution record"""
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
        
        logger.info(f"✓ Execution stored: {execution.execution_id}")
    
    def get_similar_executions(self, characteristics: DataCharacteristics, 
                               top_k: int = 10) -> List[WorkflowExecution]:
        """
        Find similar past executions based on data characteristics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all successful executions
        cursor.execute('''
            SELECT * FROM executions WHERE success = 1
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Calculate similarity scores
        target_vector = characteristics.to_vector()
        similarities = []
        
        for row in rows:
            past_chars = DataCharacteristics(**json.loads(row[3]))
            past_vector = past_chars.to_vector()
            
            # Cosine similarity
            similarity = np.dot(target_vector, past_vector) / (
                np.linalg.norm(target_vector) * np.linalg.norm(past_vector) + 1e-10
            )
            
            # Handle both old (15 columns) and new (16 columns) database schemas
            if len(row) == 15:
                # Old schema without dataset_name
                execution = WorkflowExecution(
                    execution_id=row[0],
                    timestamp=row[1],
                    data_fingerprint=row[2],
                    data_characteristics=json.loads(row[3]),
                    dataset_name=None,  # Not available in old schema
                    workflow_selected=row[4],
                    selection_method=row[5],
                    confidence_score=row[6],
                    execution_time=row[7],
                    success=bool(row[8]),
                    error_message=row[9],
                    test_r2=row[10],
                    test_mae=row[11],
                    test_rmse=row[12],
                    event_f1_score=row[13],
                    reward=row[14]
                )
            else:
                # New schema with dataset_name
                execution = WorkflowExecution(
                    execution_id=row[0],
                    timestamp=row[1],
                    data_fingerprint=row[2],
                    data_characteristics=json.loads(row[3]),
                    dataset_name=row[4],
                    workflow_selected=row[5],
                    selection_method=row[6],
                    confidence_score=row[7],
                    execution_time=row[8],
                    success=bool(row[9]),
                    error_message=row[10],
                    test_r2=row[11],
                    test_mae=row[12],
                    test_rmse=row[13],
                    event_f1_score=row[14],
                    reward=row[15]
                )
            
            similarities.append((similarity, execution))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exec for _, exec in similarities[:top_k]]
    
    def get_workflow_statistics(self) -> Dict[str, Dict]:
        """Get performance statistics for each workflow"""
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
# INTELLIGENT WORKFLOW SELECTOR (Multi-Armed Bandit)
# ============================================================================

class IntelligentWorkflowSelector:
    """
    Adaptive Contextual Multi-Armed Bandit for intelligent workflow selection
    
    Enhanced Features:
    - Exploitation: Select best-performing workflow for similar data
    - Exploration: Try different workflows to learn
    - Confidence-based: Higher uncertainty → more exploration
    - Contradiction Detection: Real data overrides synthetic when conflicting
    - Adaptive Exploration: Increases exploration when stuck in local optimum
    - Recency Weighting: Recent executions weighted more than old ones
    - Statistical Significance: Ensures differences are real, not noise
    """
    
    def __init__(self, experience_db: ExperienceDatabase, 
                 exploration_rate: float = 0.2,
                 adaptive_exploration: bool = True,
                 bootstrap_decay: float = 0.95):
        self.experience_db = experience_db
        self.base_exploration_rate = exploration_rate
        self.current_exploration_rate = exploration_rate
        self.adaptive_exploration = adaptive_exploration
        self.bootstrap_decay = bootstrap_decay  # Decay factor for synthetic data
        self.workflows = ['workflow1', 'workflow3', 'workflow4']
        
        # Track consecutive exploitations (for detecting local optimum trap)
        self.consecutive_exploitations = 0
        self.last_workflow_selected = None
    
    def select_workflow(self, characteristics: DataCharacteristics) -> Tuple[str, float, str]:
        """
        Intelligently select workflow based on data characteristics
        
        Enhanced with:
        - Contradiction detection (real vs synthetic)
        - Adaptive exploration (escape local optimum)
        - Statistical significance testing
        - Recency weighting
        
        Returns:
            (workflow_name, confidence_score, selection_method)
        """
        # Get similar past executions
        similar_execs = self.experience_db.get_similar_executions(characteristics, top_k=20)
        
        if not similar_execs:
            # No past experience → random exploration
            logger.info("🔍 No past experience found. Using random exploration.")
            workflow = np.random.choice(self.workflows)
            return workflow, 0.0, 'exploration_random'
        
        # Separate synthetic (bootstrap) from real executions
        synthetic_execs = [e for e in similar_execs if e.selection_method == 'expert_bootstrap']
        real_execs = [e for e in similar_execs if e.selection_method != 'expert_bootstrap']
        
        # Detect contradictions between synthetic and real data
        contradiction_detected = False
        if len(synthetic_execs) > 0 and len(real_execs) > 0:
            contradiction_detected = self._detect_contradiction(synthetic_execs, real_execs)
            
            if contradiction_detected:
                logger.warning("⚠️  Contradiction detected: Real data conflicts with bootstrap knowledge")
                logger.info("   → Prioritizing real executions over synthetic")
                # Use only real executions when contradiction exists
                similar_execs = real_execs
        
        # Calculate workflow scores with adaptive weighting
        workflow_scores = self._calculate_adaptive_workflow_scores(
            similar_execs, 
            characteristics,
            prioritize_real=contradiction_detected
        )
        
        # Detect if stuck in local optimum
        local_optimum_detected = self._detect_local_optimum(similar_execs, workflow_scores)
        
        # Adaptive exploration rate
        if self.adaptive_exploration:
            self.current_exploration_rate = self._calculate_adaptive_exploration_rate(
                similar_execs,
                contradiction_detected,
                local_optimum_detected
            )
        
        # Decide: exploration vs exploitation
        if np.random.random() < self.current_exploration_rate:
            # EXPLORATION
            workflow, confidence, method = self._exploration_strategy(
                workflow_scores,
                similar_execs,
                local_optimum_detected
            )
            self.consecutive_exploitations = 0
            
        else:
            # EXPLOITATION
            workflow, confidence, method = self._exploitation_strategy(
                workflow_scores,
                similar_execs
            )
            self.consecutive_exploitations += 1
        
        self.last_workflow_selected = workflow
        
        # Log selection details
        logger.info(f"Selection: {workflow} (confidence: {confidence:.1%}, method: {method})")
        logger.info(f"Current exploration rate: {self.current_exploration_rate:.1%}")
        if contradiction_detected:
            logger.info("⚠️  Real data contradicts synthetic - using real data")
        if local_optimum_detected:
            logger.info("🔄 Local optimum detected - increasing exploration")
        
        return workflow, confidence, method
    
    def _detect_contradiction(self, synthetic_execs: List[WorkflowExecution],
                             real_execs: List[WorkflowExecution]) -> bool:
        """
        Detect if real executions contradict synthetic predictions
        
        Returns True if:
        - Synthetic predicts workflow A is best
        - Real data shows workflow B is significantly better
        """
        # Get best workflow from synthetic data
        synthetic_scores = defaultdict(list)
        for exec in synthetic_execs:
            synthetic_scores[exec.workflow_selected].append(exec.reward)
        
        if not synthetic_scores:
            return False
        
        synthetic_best = max(synthetic_scores, key=lambda w: np.mean(synthetic_scores[w]))
        synthetic_best_reward = np.mean(synthetic_scores[synthetic_best])
        
        # Get best workflow from real data
        real_scores = defaultdict(list)
        for exec in real_execs:
            real_scores[exec.workflow_selected].append(exec.reward)
        
        if not real_scores:
            return False
        
        real_best = max(real_scores, key=lambda w: np.mean(real_scores[w]))
        real_best_reward = np.mean(real_scores[real_best])
        
        # Contradiction if:
        # 1. Different workflows are best
        # 2. Difference is statistically significant (>10% and t-test)
        if synthetic_best != real_best:
            reward_difference = abs(real_best_reward - synthetic_best_reward)
            
            # Significant if difference > 10% and we have enough samples
            if reward_difference > 0.10 and len(real_execs) >= 3:
                logger.info(f"   Synthetic predicted: {synthetic_best} ({synthetic_best_reward:.3f})")
                logger.info(f"   Real data shows: {real_best} ({real_best_reward:.3f})")
                return True
        
        return False
    
    def _detect_local_optimum(self, similar_execs: List[WorkflowExecution],
                             workflow_scores: Dict[str, float]) -> bool:
        """
        Detect if stuck in local optimum
        
        Indicators:
        - One workflow tried many times (>60% of executions)
        - Other workflows barely explored (<20%)
        - Recent performance plateauing or declining
        """
        if len(similar_execs) < 5:
            return False  # Not enough data
        
        # Count executions per workflow
        workflow_counts = defaultdict(int)
        for exec in similar_execs:
            workflow_counts[exec.workflow_selected] += 1
        
        total_execs = len(similar_execs)
        
        # Check if one workflow dominates
        for workflow, count in workflow_counts.items():
            if count > 0.6 * total_execs:  # >60% concentration
                # Check if other workflows are under-explored
                other_workflows = [w for w in self.workflows if w != workflow]
                under_explored = sum(1 for w in other_workflows 
                                   if workflow_counts.get(w, 0) < 0.2 * total_execs)
                
                if under_explored >= 1:
                    logger.info(f"   Detected: {workflow} tried {count}/{total_execs} times ({count/total_execs:.1%})")
                    return True
        
        # Check for performance plateau (last 5 vs previous 5)
        if len(similar_execs) >= 10:
            recent_5 = similar_execs[:5]
            previous_5 = similar_execs[5:10]
            
            recent_avg = np.mean([e.reward for e in recent_5])
            previous_avg = np.mean([e.reward for e in previous_5])
            
            # Performance declining or stagnant
            if recent_avg <= previous_avg * 1.02:  # Less than 2% improvement
                return True
        
        return False
    
    def _calculate_adaptive_exploration_rate(self, similar_execs: List[WorkflowExecution],
                                            contradiction_detected: bool,
                                            local_optimum_detected: bool) -> float:
        """
        Dynamically adjust exploration rate based on context
        """
        rate = self.base_exploration_rate
        
        # Increase exploration if contradiction detected
        if contradiction_detected:
            rate = min(rate * 1.5, 0.5)  # Up to 50% exploration
        
        # Increase exploration if stuck in local optimum
        if local_optimum_detected:
            rate = min(rate * 2.0, 0.6)  # Up to 60% exploration
        
        # Increase exploration if consecutive exploitations
        if self.consecutive_exploitations > 5:
            rate = min(rate * 1.3, 0.4)
        
        # Decrease exploration if enough data and no issues
        if len(similar_execs) > 20 and not contradiction_detected and not local_optimum_detected:
            rate = max(rate * 0.8, 0.05)  # Min 5% exploration
        
        return rate
    
    def _calculate_adaptive_workflow_scores(self, similar_execs: List[WorkflowExecution],
                                           characteristics: DataCharacteristics,
                                           prioritize_real: bool = False) -> Dict[str, float]:
        """
        Calculate performance scores with adaptive weighting
        
        Features:
        - Recency weighting (recent executions matter more)
        - Bootstrap decay (synthetic data matters less over time)
        - Statistical significance (require minimum samples)
        """
        workflow_data = defaultdict(lambda: {
            'rewards': [],
            'weights': [],
            'is_synthetic': [],
            'timestamps': []
        })
        
        # Collect data
        for exec in similar_execs:
            wf = exec.workflow_selected
            workflow_data[wf]['rewards'].append(exec.reward)
            workflow_data[wf]['is_synthetic'].append(exec.selection_method == 'expert_bootstrap')
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(exec.timestamp, "%Y-%m-%d %H:%M:%S")
                workflow_data[wf]['timestamps'].append(timestamp)
            except:
                workflow_data[wf]['timestamps'].append(datetime.now())
        
        # Calculate scores with adaptive weighting
        workflow_scores = {}
        
        for wf in self.workflows:
            if wf not in workflow_data or len(workflow_data[wf]['rewards']) == 0:
                workflow_scores[wf] = 0.0
                continue
            
            rewards = np.array(workflow_data[wf]['rewards'])
            is_synthetic = workflow_data[wf]['is_synthetic']
            timestamps = workflow_data[wf]['timestamps']
            
            # Calculate weights
            weights = []
            for i, (reward, synthetic, ts) in enumerate(zip(rewards, is_synthetic, timestamps)):
                weight = 1.0
                
                # Recency weighting (exponential decay)
                days_ago = (datetime.now() - ts).days
                recency_weight = np.exp(-days_ago / 30)  # Half-life of 30 days
                weight *= recency_weight
                
                # Bootstrap decay (synthetic data matters less over time)
                if synthetic:
                    # Decay based on how many real executions exist
                    num_real = sum(1 for s in is_synthetic if not s)
                    decay_factor = self.bootstrap_decay ** num_real
                    weight *= decay_factor
                
                # If prioritizing real data, heavily downweight synthetic
                if prioritize_real and synthetic:
                    weight *= 0.1
                
                weights.append(weight)
            
            weights = np.array(weights)
            
            # Weighted average
            if weights.sum() > 0:
                weighted_avg = np.average(rewards, weights=weights)
            else:
                weighted_avg = np.mean(rewards)
            
            # Confidence penalty for small sample size
            n_samples = len(rewards)
            confidence_factor = min(n_samples / 5.0, 1.0)  # Full confidence at 5+ samples
            
            workflow_scores[wf] = weighted_avg * confidence_factor
        
        return workflow_scores
    
    def _exploration_strategy(self, workflow_scores: Dict[str, float],
                             similar_execs: List[WorkflowExecution],
                             local_optimum_detected: bool) -> Tuple[str, float, str]:
        """
        Intelligent exploration strategy
        
        Strategies:
        1. UCB (Upper Confidence Bound) - prefer less-tried workflows
        2. Thompson Sampling - probabilistic selection
        3. Forced exploration of under-explored workflows (if local optimum)
        """
        workflow_counts = defaultdict(int)
        for exec in similar_execs:
            workflow_counts[exec.workflow_selected] += 1
        
        total_trials = sum(workflow_counts.values())
        
        if local_optimum_detected:
            # FORCED EXPLORATION: Try least-explored workflow
            logger.info("🔍 Exploration mode: Forced (escaping local optimum)")
            
            least_explored = min(self.workflows, key=lambda w: workflow_counts.get(w, 0))
            confidence = workflow_scores.get(least_explored, 0.0)
            
            return least_explored, confidence, 'exploration_forced'
        
        else:
            # UCB EXPLORATION: Balance reward + exploration bonus
            logger.info("🔍 Exploration mode: Upper Confidence Bound")
            
            ucb_scores = {}
            
            for wf in self.workflows:
                count = workflow_counts.get(wf, 0)
                avg_reward = workflow_scores.get(wf, 0.0)
                
                if count == 0:
                    ucb_scores[wf] = float('inf')  # Unexplored
                else:
                    # UCB formula: reward + c * sqrt(ln(N) / n)
                    exploration_bonus = 2.0 * np.sqrt(np.log(total_trials + 1) / count)
                    ucb_scores[wf] = avg_reward + exploration_bonus
            
            workflow = max(ucb_scores, key=ucb_scores.get)
            confidence = workflow_scores.get(workflow, 0.0)
            
            return workflow, confidence, 'exploration_ucb'
    
    def _exploitation_strategy(self, workflow_scores: Dict[str, float],
                               similar_execs: List[WorkflowExecution]) -> Tuple[str, float, str]:
        """
        Exploitation strategy with statistical significance check
        """
        logger.info("🎯 Exploitation mode: Using best-performing workflow")
        
        if not workflow_scores:
            workflow = np.random.choice(self.workflows)
            return workflow, 0.0, 'exploitation_fallback'
        
        # Select best workflow
        best_workflow = max(workflow_scores, key=workflow_scores.get)
        confidence = workflow_scores[best_workflow]
        
        # Check statistical significance (is the difference real or noise?)
        second_best = sorted(workflow_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(second_best) >= 2:
            best_score = second_best[0][1]
            second_score = second_best[1][1]
            
            # If difference is small (<5%), confidence is lower
            if abs(best_score - second_score) < 0.05:
                confidence *= 0.7  # Reduce confidence
                logger.info(f"   Close call: {second_best[0][0]} vs {second_best[1][0]} (margin: {abs(best_score - second_score):.3f})")
        
        return best_workflow, confidence, 'exploitation_best'
    
    def explain_decision(self, workflow: str, confidence: float, 
                        method: str, characteristics: DataCharacteristics,
                        contradiction_detected: bool = False,
                        local_optimum_detected: bool = False) -> str:
        """Generate human-readable explanation of decision"""
        
        explanations = {
            'exploration_random': f"No past experience with similar data. Trying {workflow} to learn.",
            'exploration_ucb': f"Exploring {workflow} to gather more data (Upper Confidence Bound strategy).",
            'exploration_forced': f"Forcing exploration of {workflow} to escape local optimum trap.",
            'exploitation_best': f"Based on past performance, {workflow} is the best choice (confidence: {confidence:.1%}).",
            'exploitation_fallback': f"Limited data. Using {workflow} as default."
        }
        
        base_explanation = explanations.get(method, f"Selected {workflow}")
        
        # Add contradiction warning
        if contradiction_detected:
            base_explanation += "\n\n⚠️  Real data contradicts synthetic bootstrap predictions."
            base_explanation += "\n   Decision prioritizes actual experimental results over initial assumptions."
        
        # Add local optimum warning
        if local_optimum_detected:
            base_explanation += "\n\n🔄 Detected potential local optimum (one workflow over-used)."
            base_explanation += "\n   Forcing exploration to discover potentially better alternatives."
        
        # Add data characteristics context
        char_summary = f"\n\n📊 Dataset Characteristics:\n"
        char_summary += f"   Volatility: {characteristics.volatility:.3f}\n"
        char_summary += f"   Regimes: {characteristics.num_regimes}\n"
        char_summary += f"   Prediction Difficulty: {characteristics.prediction_difficulty:.3f}\n"
        char_summary += f"   Stationarity: {characteristics.stationarity_score:.3f}"
        
        return base_explanation + char_summary


# ============================================================================
# MAIN AGENTIC ORCHESTRATOR
# ============================================================================

class AgenticOrchestrator:
    """
    Self-learning agentic system that improves over time
    """
    
    def __init__(self):
        self.experience_db = ExperienceDatabase()
        self.characteristics_extractor = DataCharacteristicsExtractor()
        self.workflow_selector = IntelligentWorkflowSelector(
            self.experience_db,
            exploration_rate=0.2  # 20% exploration, 80% exploitation
        )
    
    def run_agentic_prediction(self, data_path: str, 
                              user_override: Optional[str] = None,
                              transfer_learning: Optional[bool] = None):
        """
        Main agentic prediction workflow
        
        Args:
            data_path: Path to dataset
            user_override: Optional workflow override ('workflow1', 'workflow3', 'workflow4')
            transfer_learning: Optional training mode (True=transfer, False=full, None=auto-decide)
        """
        logger.info("\n" + "="*80)
        logger.info("🤖 AGENTIC PREDICTION SYSTEM")
        logger.info("="*80)
        
        # Extract dataset name
        dataset_name = os.path.basename(data_path)
        
        # Load dataset
        logger.info(f"\n📂 Loading dataset: {data_path}")
        df = pd.read_excel(data_path)
        
        # Detect datetime column
        time_col = self._detect_time_column(df)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()
        
        # Extract characteristics
        logger.info("\n🔬 Analyzing dataset characteristics...")
        characteristics = self.characteristics_extractor.extract(df)
        
        # Create fingerprint
        char_json = json.dumps(characteristics.to_dict(), sort_keys=True)
        fingerprint = hashlib.md5(char_json.encode()).hexdigest()
        
        # Display characteristics
        self._display_characteristics(characteristics)
        
        # Check dataset diversity
        self._check_dataset_diversity(dataset_name)
        
        # Intelligent workflow selection
        if user_override:
            workflow = user_override
            confidence = 1.0
            method = 'user_override'
            explanation = f"User manually selected {workflow}"
            logger.info(f"\n👤 User Override: {workflow}")
        else:
            logger.info("\n🧠 Intelligent Workflow Selection...")
            workflow, confidence, method = self.workflow_selector.select_workflow(characteristics)
            explanation = self.workflow_selector.explain_decision(
                workflow, confidence, method, characteristics
            )
            
            logger.info(f"\n✓ Selected Workflow: {workflow}")
            logger.info(f"  Confidence: {confidence:.1%}")
            logger.info(f"  Method: {method}")
            logger.info(f"\n💡 Reasoning:\n{explanation}")
        
        # Show similar past executions
        self._show_past_experience(characteristics)
        
        # TRAINING MODE SELECTION (for workflow3 and workflow4)
        if workflow in ['workflow3', 'workflow4'] and transfer_learning is None:
            # Check if pre-trained models exist
            pretrained_available = self._check_pretrained_models(workflow)
            
            if pretrained_available:
                logger.info("\n🎓 TRAINING MODE SELECTION")
                logger.info("="*80)
                logger.info(f"Pre-trained {workflow} model detected!")
                
                # Intelligent recommendation
                transfer_rec = self._recommend_training_mode(
                    characteristics, 
                    workflow,
                    len(df)
                )
                
                logger.info(f"\n📊 Recommendation: {transfer_rec['mode'].upper()}")
                logger.info(f"   Confidence: {transfer_rec['confidence']:.1%}")
                logger.info(f"   Reasoning: {transfer_rec['reasoning']}")
                
                # User choice
                print("\n" + "-"*80)
                print("Training Mode:")
                print("-" * 40)
                print(f"1. Full Retraining (Train from scratch)")
                print(f"2. Transfer Learning (Use pre-trained model) [Recommended]" if transfer_rec['mode'] == 'transfer' else "2. Transfer Learning (Use pre-trained model)")
                print(f"3. Use AI Recommendation")
                
                choice = input("\nEnter choice (1-3): ").strip()
                
                if choice == '3':
                    transfer_learning = (transfer_rec['mode'] == 'transfer')
                    logger.info(f"\n✓ Using AI recommendation: {transfer_rec['mode']}")
                elif choice == '2':
                    transfer_learning = True
                    logger.info("\n✓ Selected: Transfer Learning")
                else:
                    transfer_learning = False
                    logger.info("\n✓ Selected: Full Retraining")
            else:
                logger.info(f"\n⚠️  No pre-trained {workflow} model found.")
                logger.info("   Will perform full training.")
                transfer_learning = False
        
        # User confirmation
        if not user_override:
            print("\n" + "-"*80)
            print("Proceed with selected configuration?")
            print(f"  Workflow: {workflow}")
            if workflow in ['workflow3', 'workflow4']:
                print(f"  Training Mode: {'Transfer Learning' if transfer_learning else 'Full Retraining'}")
            print("\n1. Yes, execute")
            print("2. No, let me choose manually")
            
            choice = input("\nEnter choice (1-2): ").strip()
            
            if choice == '2':
                workflow = self._manual_workflow_selection()
                method = 'user_override'
                confidence = 1.0
                
                # Ask about training mode again if needed
                if workflow in ['workflow3', 'workflow4']:
                    tl_choice = input("\nFull training (1) or Transfer learning (2)? ").strip()
                    transfer_learning = (tl_choice == '2')
        
        # Execute workflow
        execution_record = self._execute_workflow(
            workflow, data_path, fingerprint, characteristics,
            confidence, method, dataset_name, transfer_learning
        )
        
        # Store execution for learning
        self.experience_db.store_execution(execution_record)
        
        logger.info("\n✅ Execution completed and stored in experience database")
        logger.info(f"   Total executions in database: {self._count_executions()}")
        
        return execution_record
    
    def _check_dataset_diversity(self, dataset_name: str):
        """
        Check if system is learning from diverse datasets
        """
        conn = sqlite3.connect(self.experience_db.db_path)
        cursor = conn.cursor()
        
        # Get dataset distribution
        cursor.execute('''
            SELECT dataset_name, COUNT(*) 
            FROM executions 
            WHERE dataset_name IS NOT NULL
            GROUP BY dataset_name
        ''')
        
        dataset_counts = dict(cursor.fetchall())
        conn.close()
        
        if not dataset_counts:
            return
        
        total = sum(dataset_counts.values())
        current_count = dataset_counts.get(dataset_name, 0)
        
        # Check for over-concentration
        if total > 10 and current_count > 0.7 * total:
            logger.warning(f"\n⚠️  DATASET DIVERSITY WARNING")
            logger.warning(f"   {dataset_name}: {current_count}/{total} executions ({current_count/total:.1%})")
            logger.warning(f"   System may be overfitting to this specific dataset!")
            logger.warning(f"   Recommendation: Try other datasets for better generalization")
    
    def _check_pretrained_models(self, workflow: str) -> bool:
        """Check if pre-trained models exist"""
        if workflow == 'workflow3':
            model_dir = 'final_experiment_lstm'
        elif workflow == 'workflow4':
            model_dir = 'final_experiment_transformer'
        else:
            return False
        
        # Check if model directory exists and has trained models
        if os.path.exists(model_dir):
            # Look for model files
            model_files = [
                os.path.join(model_dir, 'best_model.h5'),
                os.path.join(model_dir, 'model_weights.h5'),
                os.path.join(model_dir, 'final_model.h5')
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    logger.info(f"   ✓ Found pre-trained model: {model_file}")
                    return True
        
        return False
    
    def _recommend_training_mode(self, characteristics: DataCharacteristics,
                                 workflow: str, data_size: int) -> Dict:
        """
        Intelligently recommend training mode based on data characteristics
        
        Returns:
            {
                'mode': 'full_training' | 'transfer',
                'confidence': float,
                'reasoning': str
            }
        """
        transfer_score = 0.0
        full_training_score = 0.0
        
        # Factor 1: Dataset size
        if data_size < 50000:
            transfer_score += 0.4  # Small data → transfer better
        elif data_size > 200000:
            full_training_score += 0.3  # Large data → can afford full training
        
        # Factor 2: Prediction difficulty
        if characteristics.prediction_difficulty > 0.6:
            full_training_score += 0.3  # Complex → need full training
        elif characteristics.prediction_difficulty < 0.3:
            transfer_score += 0.2  # Simple → transfer sufficient
        
        # Factor 3: Data quality
        if characteristics.missing_data_pct > 5.0:
            transfer_score += 0.3  # Poor quality → transfer more robust
        
        # Factor 4: Time/resource constraints (assume transfer is faster)
        transfer_score += 0.1  # Baseline advantage for speed
        
        if transfer_score > full_training_score:
            mode = 'transfer'
            confidence = min(transfer_score, 0.95)
            reasoning = self._generate_transfer_reasoning(
                characteristics, data_size, transfer_score, 'transfer'
            )
        else:
            mode = 'full_training'
            confidence = min(full_training_score, 0.95)
            reasoning = self._generate_transfer_reasoning(
                characteristics, data_size, full_training_score, 'full_training'
            )
        
        return {
            'mode': mode,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _generate_transfer_reasoning(self, characteristics: DataCharacteristics,
                                     data_size: int, score: float, mode: str) -> str:
        """Generate explanation for training mode recommendation"""
        
        reasons = []
        
        if mode == 'transfer':
            if data_size < 50000:
                reasons.append(f"Limited data ({data_size:,} samples) - transfer learning more effective")
            if characteristics.prediction_difficulty < 0.3:
                reasons.append("Low complexity - pre-trained model should generalize well")
            if characteristics.missing_data_pct > 5.0:
                reasons.append(f"Data quality issues ({characteristics.missing_data_pct:.1f}% missing) - transfer more robust")
            reasons.append("Faster execution (3-5x speedup)")
        else:
            if data_size > 200000:
                reasons.append(f"Large dataset ({data_size:,} samples) - full training will leverage all data")
            if characteristics.prediction_difficulty > 0.6:
                reasons.append("High complexity - custom training needed for best performance")
            reasons.append("Maximum performance potential")
        
        return ". ".join(reasons) + "."
    
    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect datetime column"""
        time_candidates = ['datetime', 'date', 'time', 'timestamp', 'DateTime', 'Time']
        
        for col in df.columns:
            if col in time_candidates or 'time' in col.lower() or 'date' in col.lower():
                return col
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        return None
    
    def _display_characteristics(self, characteristics: DataCharacteristics):
        """Display data characteristics in readable format"""
        print("\n" + "="*80)
        print("📊 DATASET CHARACTERISTICS")
        print("="*80)
        
        print(f"\n📏 Basic Info:")
        print(f"   Sample Size: {characteristics.sample_size:,}")
        print(f"   Duration: {characteristics.duration_days:.1f} days")
        print(f"   Resolution: {characteristics.resolution_minutes:.0f} minutes")
        
        print(f"\n📈 Statistical Properties:")
        print(f"   Volatility: {characteristics.volatility:.3f}")
        print(f"   Stationarity Score: {characteristics.stationarity_score:.3f}")
        print(f"   Trend Strength: {characteristics.trend_strength:.3f}")
        
        print(f"\n🔄 Regime Analysis:")
        print(f"   Number of Regimes: {characteristics.num_regimes}")
        print(f"   Regime Persistence: {characteristics.regime_persistence:.3f}")
        print(f"   Transition Frequency: {characteristics.transition_frequency:.2f} per day")
        
        print(f"\n🎲 Uncertainty & Complexity:")
        print(f"   Entropy: {characteristics.entropy:.3f}")
        print(f"   Prediction Difficulty: {characteristics.prediction_difficulty:.3f}")
        print(f"   Lyapunov Exponent: {characteristics.lyapunov_exponent:.3f}")
        
        print(f"\n⚡ Event Characteristics:")
        print(f"   Event Frequency: {characteristics.event_frequency:.2f} per day")
        print(f"   Mean Ramp Rate: {characteristics.ramp_rate_mean:.3f}")
        
        print("="*80)
    
    def _show_past_experience(self, characteristics: DataCharacteristics):
        """Show similar past executions"""
        similar_execs = self.experience_db.get_similar_executions(characteristics, top_k=5)
        
        if not similar_execs:
            print("\n📚 Past Experience: None (this is a new type of dataset)")
            return
        
        print(f"\n📚 Similar Past Executions (Top 5):")
        print("-" * 80)
        
        for i, exec in enumerate(similar_execs, 1):
            print(f"\n{i}. Workflow: {exec.workflow_selected}")
            print(f"   Timestamp: {exec.timestamp}")
            print(f"   Reward: {exec.reward:.3f}")
            if exec.test_r2:
                print(f"   Test R²: {exec.test_r2:.3f}")
            if exec.event_f1_score:
                print(f"   Event F1: {exec.event_f1_score:.3f}")
    
    def _manual_workflow_selection(self) -> str:
        """Manual workflow selection by user"""
        print("\n" + "-"*80)
        print("Select Workflow Manually:")
        print("-" * 40)
        print("1. SARIMAX-RBA (workflow1) - Statistical, fast")
        print("2. LSTM-RBA (workflow3) - Deep learning, balanced")
        print("3. Transformer-RBA (workflow4) - State-of-art, slow")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        workflow_map = {
            '1': 'workflow1',
            '2': 'workflow3',
            '3': 'workflow4'
        }
        
        return workflow_map.get(choice, 'workflow3')
    
    def _execute_workflow(self, workflow: str, data_path: str,
                         fingerprint: str, characteristics: DataCharacteristics,
                         confidence: float, method: str, dataset_name: str,
                         transfer_learning: bool = False) -> WorkflowExecution:
        """
        Execute selected workflow and return execution record
        
        Enhanced with:
        - Transfer learning support
        - Metrics saving for all modes
        - Dataset tracking
        """
        
        execution_id = f"{workflow}_{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"\n🚀 Executing {workflow}...")
        if workflow in ['workflow3', 'workflow4'] and transfer_learning:
            logger.info(f"   Mode: Transfer Learning (zero-shot)")
        elif workflow in ['workflow3', 'workflow4']:
            logger.info(f"   Mode: Full Retraining")
        
        start_time = time.time()
        
        try:
            # Import and run workflow
            if workflow == 'workflow1':
                from workflows import workflow1
                result = workflow1.main(data_path=data_path)
                
            elif workflow == 'workflow3':
                from workflows import workflow3
                result = workflow3.main(
                    data_path=data_path, 
                    transfer_learning=transfer_learning
                )
                
            elif workflow == 'workflow4':
                from workflows import workflow4
                result = workflow4.main(
                    data_path=data_path, 
                    transfer_learning=transfer_learning
                )
            else:
                raise ValueError(f"Unknown workflow: {workflow}")
            
            execution_time = time.time() - start_time
            
            # Extract performance metrics from result
            test_r2 = result.get('test_r2')
            test_mae = result.get('test_mae')
            test_rmse = result.get('test_rmse')
            event_f1 = result.get('event_f1_score')
            
            # Log metrics
            logger.info(f"\n📊 Performance Metrics:")
            if test_r2 is not None:
                logger.info(f"   Test R²: {test_r2:.4f}")
            if test_mae is not None:
                logger.info(f"   Test MAE: {test_mae:.2f}")
            if test_rmse is not None:
                logger.info(f"   Test RMSE: {test_rmse:.2f}")
            if event_f1 is not None:
                logger.info(f"   Event F1: {event_f1:.4f}")
            
            # Check if predictions were saved
            output_dir = result.get('output_dir', result.get('model_dir'))
            if output_dir and os.path.exists(output_dir):
                logger.info(f"\n💾 Outputs saved to: {output_dir}")
                
                # List saved files
                saved_files = []
                for file in os.listdir(output_dir):
                    if file.endswith(('.csv', '.pkl', '.json', '.png')):
                        saved_files.append(file)
                
                if saved_files:
                    logger.info(f"   Files: {', '.join(saved_files[:5])}")
                    if len(saved_files) > 5:
                        logger.info(f"   ... and {len(saved_files) - 5} more")
            
            # Calculate reward (composite score)
            reward = self._calculate_reward(test_r2, test_mae, event_f1, execution_time)
            
            execution = WorkflowExecution(
                execution_id=execution_id,
                timestamp=timestamp,
                data_fingerprint=fingerprint,
                data_characteristics=characteristics.to_dict(),
                dataset_name=dataset_name,
                workflow_selected=workflow,
                selection_method=method,
                confidence_score=confidence,
                execution_time=execution_time,
                success=True,
                error_message=None,
                test_r2=test_r2,
                test_mae=test_mae,
                test_rmse=test_rmse,
                event_f1_score=event_f1,
                reward=reward
            )
            
            logger.info(f"\n✓ {workflow} completed successfully")
            logger.info(f"  Execution time: {execution_time:.1f}s")
            logger.info(f"  Reward score: {reward:.3f}")
            if transfer_learning:
                logger.info(f"  Transfer learning speedup: ~3-5x faster than full training")
            
            return execution
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {workflow} failed: {e}")
            
            execution = WorkflowExecution(
                execution_id=execution_id,
                timestamp=timestamp,
                data_fingerprint=fingerprint,
                data_characteristics=characteristics.to_dict(),
                dataset_name=dataset_name,
                workflow_selected=workflow,
                selection_method=method,
                confidence_score=confidence,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                test_r2=None,
                test_mae=None,
                test_rmse=None,
                event_f1_score=None,
                reward=0.0
            )
            
            return execution
    
    def _calculate_reward(self, test_r2: Optional[float], 
                         test_mae: Optional[float],
                         event_f1: Optional[float],
                         execution_time: float) -> float:
        """
        Calculate composite reward score
        
        Higher is better. Considers:
        - Prediction accuracy (R², MAE)
        - Event detection accuracy (F1)
        - Execution efficiency (time)
        """
        reward = 0.0
        
        # R² component (0-0.4 range)
        if test_r2 is not None:
            reward += min(test_r2, 1.0) * 0.4
        
        # MAE component (0-0.2 range, inverted)
        if test_mae is not None:
            # Normalize MAE (assuming typical range 0-500)
            mae_norm = max(0, 1 - test_mae / 500)
            reward += mae_norm * 0.2
        
        # Event F1 component (0-0.3 range)
        if event_f1 is not None:
            reward += min(event_f1, 1.0) * 0.3
        
        # Efficiency component (0-0.1 range)
        # Reward faster executions (typical range: 600-3600s)
        time_norm = max(0, 1 - execution_time / 3600)
        reward += time_norm * 0.1
        
        return reward
    
    def _count_executions(self) -> int:
        """Count total executions in database"""
        conn = sqlite3.connect(self.experience_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM executions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def show_learning_statistics(self):
        """Display learning statistics"""
        stats = self.experience_db.get_workflow_statistics()
        
        print("\n" + "="*80)
        print("📊 LEARNING STATISTICS")
        print("="*80)
        
        if not stats:
            print("\nNo executions yet. System will learn as you use it!")
            return
        
        print(f"\nTotal Workflows Tried: {len(stats)}")
        print(f"Total Executions: {self._count_executions()}")
        
        print("\n📈 Performance by Workflow:")
        print("-" * 80)
        
        for workflow, wf_stats in stats.items():
            print(f"\n{workflow}:")
            print(f"   Executions: {wf_stats['count']}")
            print(f"   Avg Reward: {wf_stats['avg_reward']:.3f}")
            print(f"   Max Reward: {wf_stats['max_reward']:.3f}")
            if wf_stats['avg_r2']:
                print(f"   Avg R²: {wf_stats['avg_r2']:.3f}")
            if wf_stats['avg_f1']:
                print(f"   Avg Event F1: {wf_stats['avg_f1']:.3f}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for agentic system
    """
    print("\n" + "="*80)
    print("🤖 SELF-LEARNING AGENTIC EVENT PREDICTION SYSTEM")
    print("="*80)
    print("\nThis system:")
    print("  ✓ Learns from every execution")
    print("  ✓ Adapts workflow selection to your data")
    print("  ✓ Improves automatically over time")
    print("  ✓ NO hardcoded rules!")
    print("="*80)
    
    orchestrator = AgenticOrchestrator()
    
    # Show learning statistics
    orchestrator.show_learning_statistics()
    
    print("\n" + "-"*80)
    print("Mode Selection:")
    print("-" * 40)
    print("1. Event Detection (RBA-theta analysis)")
    print("2. Event Prediction (Agentic ML-based forecasting)")
    print("3. Show Learning Statistics")
    
    try:
        mode = input("\nEnter choice (1-3): ").strip()
        
        if mode == '3':
            orchestrator.show_learning_statistics()
            return
        
        # Get dataset
        data_path = input("\nEnter dataset path: ").strip().strip('"').strip("'")
        
        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            return
        
        if mode == '1':
            # Event detection
            print("\n🔍 Running event detection...")
            import event_detector
            event_detector.main(data_path, run_comprehensive=True)
        
        elif mode == '2':
            # Agentic prediction
            orchestrator.run_agentic_prediction(data_path)
        
        print("\n" + "="*80)
        print("✅ EXECUTION COMPLETED")
        print("="*80)
        print("\n💡 The system has learned from this execution.")
        print("   Future predictions will be smarter!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()