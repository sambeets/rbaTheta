#classic_model.py
"""The classic rbaTheta model"""
#from giddy.markov import LISA_Markov, Spatial_Markov
#from libpysal.weights import Queen, DistanceBand
#import libpysal
#import geopandas
import core.helpers as fn
import core.event_extraction as ee
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def RBA_theta(data, nominal, s=0.01, k=3, fc=0.3, threshold=0.15):
    """
    Args:
        data: discrete wind power (/hour) in Watts, from N turbines
        nominal: nominal production in Watts
        sampling_time: 1 for per hour, 6 for per 10 minutes, ...
        s: smoothness factor of Bspline
        k: degree of Bspline
        fc: cutoff frequency for Blackman filter-
        threshold:
    Returns:
        dictionary of [dataframe of events per turbines] x turbines
    """

    N = len(data.columns)
    turbines = []
    for i in range(1, N + 1):
        turbines.append(f'Turbine_{i}') #column names as WT1,WT2,WT3,...

    normalized_data, filtered_data = {}, {}

    for i in range(N):
        normalized_data[turbines[i]] = fn.normalize(data=data.iloc[:, i].values, nominal=nominal)
        # splined.iloc[:,i]=bspline(wind_cf.iloc[:,i],s,k)    #splined=smoothed input_data with bspline in
        #filtered_data[turbines[i]] = fn.filter_blackman(data=normalized_data[turbines[i]], fc=fc)

    tao = len(normalized_data) + 1
    significant_events, stationary_events = {}, {}

    for i in range(N):
        # sampling_time
        significant_events[turbines[i]] = ee.significant_events(data=normalized_data[turbines[i]], threshold=threshold)
        stationary_events[turbines[i]] = ee.stationary_events(data=normalized_data[turbines[i]], threshold=threshold)

    # Convert to DataFrame format for metrics
    def convert_to_dataframe(event_dict):
        if not event_dict or all(df.empty for df in event_dict.values()):
            return pd.DataFrame()
        valid_events = {k: v for k, v in event_dict.items() if not v.empty}
        if valid_events:
            return pd.concat(valid_events.values(), keys=valid_events.keys())
        return pd.DataFrame()

    sig_events_df = convert_to_dataframe(significant_events)
    stat_events_df = convert_to_dataframe(stationary_events)

    # Calculate quality metrics (same as enhanced version)
    metrics = calculate_quality_metrics(sig_events_df, stat_events_df)

    # Print results
    logger.info("=== CLASSICAL RBA_THETA QUALITY METRICS ===")
    if metrics['significant_events']:
        logger.info(f"  Significant Events Quality: {metrics['significant_events']['overall_quality_score']:.3f}")
    if metrics['stationary_events']:
        logger.info(f"  Stationary Events Quality: {metrics['stationary_events']['overall_quality_score']:.3f}")
    logger.info(f"  Balance Score: {metrics['overall']['balance_score']:.3f}")
    logger.info(f"  Total Events: {metrics['overall']['total_events']}")

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
    return [significant_events, stationary_events, tao]

def calculate_quality_metrics(sig_events_df, stat_events_df):
    """
    Same quality metrics as enhanced version - simplified
    """
    metrics = {
        'significant_events': {},
        'stationary_events': {},
        'overall': {}
    }

    # Significant Events Quality
    if not sig_events_df.empty and '∆w_m' in sig_events_df.columns:
        durations = sig_events_df['∆t_m'].values
        magnitudes = sig_events_df['∆w_m'].abs().values
        
        # 1. Ramp Consistency
        try:
            correlation = np.corrcoef(durations, magnitudes)[0, 1] if len(durations) > 1 else 0.5
            ramp_consistency = max(0, min(1, (correlation + 1) / 2)) if not np.isnan(correlation) else 0.5
        except:
            ramp_consistency = 0.5

        # 2. Magnitude Distribution
        if len(magnitudes) >= 3:
            median_mag = np.median(magnitudes)
            small_events = np.sum(magnitudes <= median_mag)
            large_events = np.sum(magnitudes > median_mag)
            ratio_score = min(1.0, small_events / (large_events + 1) / 2.0)
            cv = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0
            cv_score = max(0, min(1, 1 - abs(cv - 0.5) / 0.5))
            magnitude_score = (ratio_score + cv_score) / 2
        else:
            magnitude_score = 0.5

        # 3. Duration Reasonableness
        reasonable_events = np.sum((durations >= 2) & (durations <= 50))
        duration_score = reasonable_events / len(durations) if len(durations) > 0 else 0

        # 4. Direction Classification
        if 'θ_m' in sig_events_df.columns:
            delta_w = sig_events_df['∆w_m'].values
            angles = sig_events_df['θ_m'].values
            correct = sum(1 for dw, angle in zip(delta_w, angles) 
                         if not (np.isnan(angle) or np.isnan(dw)) and 
                         ((dw > 0 and angle > 0) or (dw < 0 and angle < 0)))
            total = sum(1 for dw, angle in zip(delta_w, angles) 
                       if not (np.isnan(angle) or np.isnan(dw)))
            direction_accuracy = correct / total if total > 0 else 0.5
        else:
            direction_accuracy = 0.0

        metrics['significant_events'] = {
            'total_events': len(sig_events_df),
            'ramp_consistency_score': ramp_consistency,
            'magnitude_distribution_score': magnitude_score,
            'duration_reasonableness_score': duration_score,
            'direction_classification_accuracy': direction_accuracy,
            'overall_quality_score': np.mean([ramp_consistency, magnitude_score, duration_score, direction_accuracy])
        }

    # Stationary Events Quality
    if not stat_events_df.empty and 'σ_s' in stat_events_df.columns:
        sigmas = stat_events_df['σ_s'].values
        durations = stat_events_df['∆t_s'].values

        # 1. Stability Consistency
        if len(sigmas) > 0:
            mean_sigma = np.mean(sigmas)
            sigma_consistency = np.std(sigmas) / mean_sigma if mean_sigma > 0 else 1
            consistency_score = max(0, min(1, 1 - sigma_consistency))
            small_sigma_ratio = np.sum(sigmas < 0.2) / len(sigmas)
            stability_score = (consistency_score + small_sigma_ratio) / 2
        else:
            stability_score = 0.0

        # 2. Duration Appropriateness
        if len(durations) > 0:
            reasonable_events = np.sum((durations >= 5) & (durations <= 200))
            appropriateness_score = reasonable_events / len(durations)
            long_periods = np.sum(durations >= 20)
            long_period_bonus = min(0.2, long_periods / len(durations))
            stat_duration_score = min(1.0, appropriateness_score + long_period_bonus)
        else:
            stat_duration_score = 0.0

        # 3. Non-overlap Score
        if 't1' in stat_events_df.columns and 't2' in stat_events_df.columns and len(stat_events_df) > 1:
            events = stat_events_df[['t1', 't2']].values
            events = events[events[:, 0].argsort()]
            overlaps = 0
            total_pairs = 0
            for i in range(len(events) - 1):
                for j in range(i + 1, len(events)):
                    total_pairs += 1
                    if not (events[j][0] >= events[i][1] or events[i][0] >= events[j][1]):
                        overlaps += 1
            non_overlap_score = 1.0 - (overlaps / total_pairs) if total_pairs > 0 else 1.0
        else:
            non_overlap_score = 1.0

        metrics['stationary_events'] = {
            'total_events': len(stat_events_df),
            'stability_consistency_score': stability_score,
            'duration_appropriateness_score': stat_duration_score,
            'non_overlap_score': non_overlap_score,
            'overall_quality_score': np.mean([stability_score, stat_duration_score, non_overlap_score])
        }

    # Overall Balance
    total_events = len(sig_events_df) + len(stat_events_df)
    sig_ratio = len(sig_events_df) / total_events if total_events > 0 else 0
    balance_score = 1.0 - abs(sig_ratio - 0.5) * 2

    metrics['overall'] = {
        'total_events': total_events,
        'significant_ratio': sig_ratio,
        'stationary_ratio': 1 - sig_ratio,
        'balance_score': balance_score
    }

    return metrics


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
    return [significant_events, stationary_events, tao]



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

