"""
RBA_theta searches for variations in a dataset within and above a given Threshold. 
"""   
"""
Enhanced event_extraction.py - Simplified version
Key improvements: Adaptive parameters, better event detection, quality filtering
Maintains ALL original function signatures
"""         
import pandas as pd
import numpy as np
from itertools import compress
from core.helpers import lam
from typing import Union, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def significant_events(data: Union[np.ndarray, List], threshold: float, 
                      min_duration: int = 3, min_slope: float = 0.05,
                      window_minutes: int = 60, freq_secs: int = 3600) -> pd.DataFrame:
    """
    Improved significant events detection with better ramp classification
    """
    
    if len(data) == 0:
        return pd.DataFrame(columns=['t1', 't2', '∆t_m', 'w_m(t1)', 'w_m(t2)', '∆w_m', 'σ_m', 'θ_m'])
    
    data = np.array(data)
    
    # Enhanced preprocessing
    data_std = np.std(data)
    data_mean = np.mean(data)
    
    # Adaptive threshold with improved sensitivity
    adaptive_threshold = threshold * (1.0 + (data_std / data_mean) * 0.15)
    
    # Improved ramp detection with better classification
    events = _detect_ramps_improved(data, adaptive_threshold, min_duration, min_slope, window_minutes)
    
    # Convert to DataFrame
    if events:
        df = pd.DataFrame(events, columns=['t1', 't2', '∆t_m', 'w_m(t1)', 'w_m(t2)', '∆w_m', 'σ_m', 'θ_m'])
        df = _filter_significant_events_improved(df, data)
        return df.reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['t1', 't2', '∆t_m', 'w_m(t1)', 'w_m(t2)', '∆w_m', 'σ_m', 'θ_m'])

def stationary_events(data: Union[np.ndarray, List], threshold: float,
                     min_duration: int = 3, min_stationary_length: int = 7,
                     window_minutes: int = 60, freq_secs: int = 3600) -> pd.DataFrame:
    """
    Refined stationary events detection - reduced over-extraction
    """
    
    if len(data) == 0:
        return pd.DataFrame(columns=['t1', 't2', '∆t_s', 'σ_s'])
    
    data = np.array(data)
    
    # DEBUG: Log the input threshold
    logger.info(f"Stationary detection - Input threshold: {threshold:.6f}")
    
    data_std = np.std(data)
    data_mean = np.mean(data)
    
    # Calculate a reasonable stationary threshold - STRICTER than before
    base_stationary_threshold = data_std * 0.15  # Increased from 0.1 to 0.15
    
    # Make it more selective
    if threshold < data_std * 0.001:  # If threshold is extremely small
        effective_threshold = base_stationary_threshold
        logger.info(f"Using data-driven threshold: {effective_threshold:.6f} (input was too small)")
    else:
        effective_threshold = max(threshold, base_stationary_threshold * 0.7)  # Increased from 0.5
        logger.info(f"Using adjusted threshold: {effective_threshold:.6f}")
    
    # REFINED stationary detection with anti-ramp checking
    events = _detect_stationary_refined(data, effective_threshold, min_stationary_length)
    
    logger.info(f"Stationary detection found {len(events)} raw events")
    
    # Convert to DataFrame with STRICTER filtering
    if events:
        df = pd.DataFrame(events, columns=['t1', 't2', '∆t_s', 'σ_s'])
        df = _filter_stationary_refined(df, data)
        logger.info(f"After filtering: {len(df)} stationary events")
        return df.reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['t1', 't2', '∆t_s', 'σ_s'])

# ==================== SIGNIFICANT EVENT HELPERS ====================

def _detect_ramps_improved(data, threshold, min_duration, min_slope, window_minutes):
    """
    Improved ramp detection with better direction classification
    """
    events = []
    n = len(data)
    
    if n < min_duration:
        return events
    
    # Apply light smoothing to reduce noise but preserve ramps
    smoothed_data = _light_smooth(data, window=3)
    
    # Calculate gradients for trend detection
    gradients = np.gradient(smoothed_data)
    
    # Find potential ramp start points
    significant_gradient_mask = np.abs(gradients) > min_slope
    potential_starts = np.where(significant_gradient_mask)[0]
    
    i = 0
    while i < len(potential_starts):
        start_idx = potential_starts[i]
        
        # Look for ramp end within reasonable window
        max_search = min(start_idx + window_minutes, n - 1)
        
        # Find the best ramp from this start point
        best_ramp = _find_best_ramp_from_start(
            smoothed_data, start_idx, max_search, min_duration, min_slope, threshold
        )
        
        if best_ramp is not None:
            events.append(best_ramp)
            # Skip ahead to avoid overlapping detections
            i = np.searchsorted(potential_starts, best_ramp[1]) + 1
        else:
            i += 1
    
    return events

def _find_best_ramp_from_start(data, start_idx, max_end, min_duration, min_slope, threshold):
    """
    Find the best ramp starting from a given point
    """
    best_ramp = None
    best_score = 0
    
    # Try different end points
    for end_idx in range(start_idx + min_duration, min(max_end + 1, len(data))):
        segment = data[start_idx:end_idx + 1]
        
        if _is_valid_ramp_segment(segment, min_slope, threshold):
            # Calculate ramp properties
            delta_t = end_idx - start_idx
            w_t1 = data[start_idx]
            w_t2 = data[end_idx]
            delta_w = w_t2 - w_t1
            sigma = np.std(segment)
            
            # Calculate angle with improved precision
            if delta_t > 0:
                theta = np.arctan2(delta_w, delta_t) * 180 / np.pi
            else:
                theta = 0.0
            
            # Score this ramp (prefer longer, more consistent ramps)
            magnitude_score = abs(delta_w)
            consistency_score = 1.0 / (1.0 + sigma)  # Lower sigma = higher score
            duration_score = delta_t
            
            total_score = magnitude_score * consistency_score * np.sqrt(duration_score)
            
            if total_score > best_score:
                best_score = total_score
                best_ramp = [start_idx, end_idx, delta_t, w_t1, w_t2, delta_w, sigma, theta]
    
    return best_ramp

def _is_valid_ramp_segment(segment, min_slope, threshold):
    """
    Check if segment qualifies as a valid ramp
    """
    if len(segment) < 3:
        return False
    
    # Check overall slope
    overall_slope = abs(segment[-1] - segment[0]) / len(segment)
    if overall_slope < min_slope:
        return False
    
    # Check magnitude
    max_change = np.max(np.abs(np.diff(segment)))
    if max_change < threshold:
        return False
    
    # Check consistency - ramp should be mostly monotonic
    diffs = np.diff(segment)
    if len(diffs) > 0:
        # Determine primary direction
        total_change = segment[-1] - segment[0]
        if abs(total_change) < threshold:
            return False
        
        primary_direction = 1 if total_change > 0 else -1
        
        # Count changes that go against the primary direction
        contrary_changes = np.sum((diffs * primary_direction) < 0)
        contrary_ratio = contrary_changes / len(diffs)
        
        # Allow some noise but reject if too many contrary changes
        if contrary_ratio > 0.4:  # More than 40% contrary changes
            return False
    
    return True

def _light_smooth(data, window=3):
    """
    Apply light smoothing to reduce noise while preserving ramps
    """
    if len(data) <= window:
        return data
    
    # Use median filter for light smoothing
    smoothed = np.copy(data)
    half_window = window // 2
    
    for i in range(half_window, len(data) - half_window):
        segment = data[i - half_window:i + half_window + 1]
        smoothed[i] = np.median(segment)
    
    return smoothed

def _filter_significant_events_improved(df, original_data):
    """
    Improved filtering for significant events
    """
    if df.empty:
        return df
    
    data_std = np.std(original_data)
    data_length = len(original_data)
    
    # More refined filters
    filters = []
    
    # Duration filter - reasonable range
    min_duration = 2
    max_duration = data_length * 0.15  # Allow longer events
    filters.append((df['∆t_m'] >= min_duration) & (df['∆t_m'] <= max_duration))
    
    # Magnitude filter - must be significant
    min_magnitude = data_std * 0.03  # Slightly lower threshold
    filters.append(df['∆w_m'].abs() >= min_magnitude)
    
    # Angle filter - reasonable slopes
    filters.append((df['θ_m'].abs() >= 0.5) & (df['θ_m'].abs() <= 89.5))
    
    # Consistency filter - sigma shouldn't be too high relative to change
    max_sigma_ratio = 0.8  # Sigma should be less than 80% of magnitude
    filters.append(df['σ_m'] <= df['∆w_m'].abs() * max_sigma_ratio)
    
    # Apply all filters
    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter & f
        return df[combined_filter]
    
    return df

#                   STATIONARY EVENT HELPERS
# ====================================================================================

def _detect_stationary_refined(data, threshold, min_length):
    """
    Refined stationary detection with anti-ramp checking and overlap prevention
    """
    events = []
    n = len(data)
    
    if n < min_length:
        return events
    
    # Use slightly larger window for better stability detection
    window_size = max(8, min_length * 1.5)  # Increased window size
    
    logger.info(f"Using refined window size: {window_size} for {n} data points")
    
    # Pre-calculate rolling statistics
    logger.info("Pre-calculating rolling statistics...")
    
    data_series = pd.Series(data)
    rolling_std = data_series.rolling(window=int(window_size), min_periods=int(window_size//2)).std()
    rolling_range = data_series.rolling(window=int(window_size), min_periods=int(window_size//2)).apply(
        lambda x: x.max() - x.min(), raw=True
    )
    
    # Calculate rolling slope to detect ramps (ANTI-RAMP check)
    rolling_slope = data_series.rolling(window=int(window_size), min_periods=int(window_size//2)).apply(
        lambda x: np.abs(np.polyfit(range(len(x)), x, 1)[0]) if len(x) >= 2 else 0, raw=True
    )
    
    # STRICTER thresholds
    std_threshold = threshold * 3.0   # Reduced from 5.0
    range_threshold = threshold * 6.0  # Reduced from 10.0
    slope_threshold = threshold * 0.5  # New: slope threshold to avoid ramps
    
    logger.info(f"Refined thresholds: std={std_threshold:.6f}, range={range_threshold:.6f}, slope={slope_threshold:.6f}")
    
    # Find stationary regions with anti-ramp checking
    stationary_mask = (
        (rolling_std <= std_threshold) & 
        (rolling_range <= range_threshold) &
        (rolling_slope <= slope_threshold)  # NEW: Must have low slope
    ).fillna(False)
    
    logger.info(f"Found {stationary_mask.sum()} potentially stationary points (after anti-ramp check)")
    
    # Find contiguous stationary regions with gap handling
    events = _find_refined_stationary_regions(stationary_mask.values, data, min_length)
    
    # Remove overlapping events
    events = _remove_overlapping_stationary_events(events)
    
    logger.info(f"Refined stationary detection found {len(events)} events")
    return events

def _find_refined_stationary_regions(mask, data, min_length):
    """
    Find stationary regions with gap tolerance and minimum separation
    """
    events = []
    n = len(mask)
    
    start = None
    gap_tolerance = 3  # Allow small gaps in stationary periods
    gap_count = 0
    
    for i in range(n):
        if mask[i]:
            if start is None:
                start = i
                gap_count = 0
            else:
                gap_count = 0  # Reset gap count
        else:
            if start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    # End of stationary region
                    end = i - gap_count
                    duration = end - start
                    if duration >= min_length:
                        sigma_s = np.std(data[start:end])
                        events.append([start, end, duration, sigma_s])
                    start = None
                    gap_count = 0
    
    # Handle case where stationary region extends to end
    if start is not None:
        end = n - gap_count if gap_count > 0 else n
        duration = end - start
        if duration >= min_length:
            sigma_s = np.std(data[start:end])
            events.append([start, end, duration, sigma_s])
    
    return events

def _remove_overlapping_stationary_events(events):
    """
    Remove overlapping stationary events, keeping the most stable ones
    """
    if not events:
        return events
    
    # Sort by start time
    events.sort(key=lambda x: x[0])
    
    filtered_events = []
    for event in events:
        start, end, duration, sigma = event
        
        # Check for overlap with existing filtered events
        overlaps = False
        for existing in filtered_events:
            existing_start, existing_end = existing[0], existing[1]
            
            # Check for any overlap
            if not (end <= existing_start or start >= existing_end):
                # There's overlap - keep the one with lower sigma (more stable)
                if sigma < existing[3]:
                    # Remove the existing event and add this one
                    filtered_events.remove(existing)
                    break
                else:
                    # Keep existing, skip this one
                    overlaps = True
                    break
        
        if not overlaps:
            filtered_events.append(event)
    
    return filtered_events

def _filter_stationary_refined(df, original_data):
    """
    Refined and stricter filtering for stationary events
    """
    if df.empty:
        return df
    
    data_std = np.std(original_data)
    data_length = len(original_data)
    
    logger.info(f"Filtering {len(df)} stationary events, data_std={data_std:.4f}")
    
    # STRICTER filters
    filters = []
    
    # Duration filter - require meaningful length
    min_duration = max(5, int(data_length * 0.001))  # At least 0.1% of data length
    max_duration = data_length * 0.3  # Reduced from 0.7
    duration_filter = (df['∆t_s'] >= min_duration) & (df['∆t_s'] <= max_duration)
    filters.append(duration_filter)
    
    logger.info(f"Duration filter (min={min_duration}): {duration_filter.sum()} events pass")
    
    # Stability filter - much stricter
    max_sigma = data_std * 0.4  # Reduced from 1.0 to 0.4
    stability_filter = df['σ_s'] <= max_sigma
    filters.append(stability_filter)
    
    logger.info(f"Stability filter (max_sigma={max_sigma:.4f}): {stability_filter.sum()} events pass")
    
    # Quality filter - remove events that are too short relative to their variability
    quality_filter = df['∆t_s'] >= (df['σ_s'] * 10)  # Duration should be 10x the sigma
    filters.append(quality_filter)
    
    logger.info(f"Quality filter: {quality_filter.sum()} events pass")
    
    # Apply all filters
    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter & f
        
        result = df[combined_filter]
        logger.info(f"Final result: {len(result)} stationary events pass all filters")
        return result
    
    return df

def rainflow(data, threshold, flm=0, l_ult=1e16, uc_mult=0.5):
    """ Rainflow counting of a signal's turning points with Goodman correction

        Args:
            array_ext (numpy.ndarray): array of turning points

        Keyword Args:
            flm (float): fixed-load mean [opt, default=0]
            l_ult (float): ultimate load [opt, default=1e16]
            uc_mult (float): partial-load scaling [opt, default=0.5]

        Returns:
            array_out (numpy.ndarray): (5 x n_cycle) array of rainflow values:
                                        1) load range
                                        2) range mean
                                        3) Goodman-adjusted range
                                        4) cycle count
                                        5) Goodman-adjusted range with flm = 0

    """
    prev = data[0]
    delta = []
    for item in data[1:]:
        delta.append(item - prev)
        prev = item

    start = 0
    maximum = [data[start], start]
    sign = lambda x: (1, -1)[x < 0]

    for i in range(len(delta) - 1):
        if sign(delta[i]) == sign(delta[i+1]):
            continue
        else:
            new_event = [data[i + 1], i + 1]
            maximum.append(new_event)
            start = i + 1

    maximum.append([data[len(data) - 1], len(data) - 1])
    
    
    #rainflow with Power range and time range of cycles
    
    l = len(maximum)                        # total size of input array
    array_out = np.zeros((l - 1, 8))        # initialize output array
    pr = 0                                  # index of input array
    po = 0                                  # index of output array
    j = -1                                  # index of temporary array "a"
    a = np.empty((len(maximum),2))          # temporary array for algorithm
    # loop through each turning point stored in input array
    for i in range(l):
        
        j += 1                  # increment "a" counter
        a[j] = maximum[pr]      # put turning point into temporary array
        pr += 1                 # increment input array pointer

        
        while ((j >= 2) & (np.fabs( a[j-1,0] - a[j-2,0] ) <= np.fabs( a[j,0] - a[j-1,0]) ) ):
            lrange = np.fabs( a[j-1,0] - a[j-2,0] )
            
            # partial range
            if j == 2:
                mean = (a[0, 0] + a[1, 0]) / 2.
                Pstart = a[0, 0]
                Pend = a[1, 0]
                tstart = a[0, 1]
                tend = a[1, 1]
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if (lrange > 0):
                    array_out[po, 0] = Pstart
                    array_out[po, 1] = Pend
                    array_out[po, 2] = tstart
                    array_out[po, 3] = tend
                    array_out[po, 4] = Pend-Pstart
                    array_out[po, 5] = tend-tstart
                    array_out[po, 6] = mean
                    array_out[po, 7] = uc_mult
                    po += 1
            # full range
            else:
                mean = (a[j-1, 0] + a[j-2, 0]) / 2.
                Pstart = a[j-2, 0]
                Pend = a[j-1, 0]
                tstart = a[j-2, 1]
                tend = a[j-1, 1]
                a[j - 2] = a[j]
                j = j - 2
                if (lrange > 0):
                    array_out[po, 0] = Pstart
                    array_out[po, 1] = Pend
                    array_out[po, 2] = tstart
                    array_out[po, 3] = tend
                    array_out[po, 4] = Pend-Pstart
                    array_out[po, 5] = tend-tstart
                    array_out[po, 6] = mean
                    array_out[po, 7] = 1.00
                    po += 1

    # partial range
    for i in range(j):
        lrange = np.fabs(a[i, 0] - a[i + 1, 0])
        mean = (a[i, 0] + a[i + 1, 0]) / 2.
        Pstart = a[i, 0]
        Pend = a[i + 1, 0]
        tstart = a[i, 1]
        tend = a[i + 1, 1]

        if (lrange > 0):
            array_out[po, 0] = Pstart
            array_out[po, 1] = Pend
            array_out[po, 2] = tstart
            array_out[po, 3] = tend
            array_out[po, 4] = Pend-Pstart
            array_out[po, 5] = tend-tstart
            array_out[po, 6] = mean
            array_out[po, 7] = uc_mult
            po += 1  
    
    # get rid of unused entries
    mask = np.ones(len(array_out), dtype=bool)
    
    for i in range(len(array_out)):
        if array_out[i, 7] == 0.0:
            mask[i]=False
    array_out = array_out[mask]
    

    angles=[0]
    
    for i in range(len(array_out)):
        n=array_out[i,4]/array_out[i,5]
        new=np.arctan(n)
        angles=np.vstack((angles,new))
    angles=angles[1:]
    angles=np.degrees(angles)
    array_out=np.c_[array_out,angles]

    """ Not in use
    angles=[]
    
    for i in range(len(array_out)):
        new=array_out[i,4]*10/array_out[i,5]
        angles.append(new)
        #else:
          #  new=math.atan2(lg[i,0],lg[i,1])
          #  angles.append(new)
    angles=np.arctan(angles)
    angles=np.degrees(angles)
    array_out=np.c_[array_out,angles]"""
    def pers(binn,array):
        freq=np.zeros((len(binn),1))
        persi=np.zeros((len(array),1))
        for k in range(len(binn)-1):
            for i in range(len(array)):
                if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                    freq[k]+=1 #how many times this(between the range of particular bin) occured

        for k in range(len(binn)-1):
            for i in range(len(array)):
                if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                    persi[i]=freq[k]
        return persi

    bins = int(1 // threshold)
    amp_bins=np.linspace(-1,1,bins)
    a=array_out[:,4]
    amp_pers=pers(amp_bins,a)
    
    angle_bins=np.linspace(-90,90,bins)
    angle_pers=pers(angle_bins,angles)
    
    max_time=max(array_out[:,5])
    time_bins=np.linspace(1,max_time,bins)
    b=array_out[:,5]
    time_pers=pers(time_bins,b)
    
    
    unique_amp=np.unique(array_out[:,4:],axis=0)
    
    #count of cycles with amplitude
    cycles=np.zeros((len(unique_amp),1))
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i,4]==unique_amp[k,0]:
                cycles[k]+=array_out[i,7]
    
    cyclecount=np.zeros([len(array_out),1])
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i,4]==unique_amp[k,0]:
                cyclecount[i]=cycles[k]
    
    #persistance=np.hstack((amp_pers,time_pers,angle_pers,cyclecount))
    array_out=np.append(array_out,amp_pers,axis=1) 
    array_out=np.append(array_out,time_pers,axis=1) 
    array_out=np.append(array_out,angle_pers,axis=1) 
    array_out=np.append(array_out,cyclecount,axis=1) 
    Array_out=pd.DataFrame(array_out,columns=['w1', 'w2', 't1', 't2', '∆w_r', '∆t_r', 'σ_r', 'cycle', 'θ_r', 'amp_freq', 'time_freq', 'angle_freq', 'cyclecount'])
    return Array_out