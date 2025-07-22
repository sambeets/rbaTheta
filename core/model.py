"""The rbaTheta model"""
#from giddy.markov import LISA_Markov, Spatial_Markov
#from libpysal.weights import Queen, DistanceBand
#import libpysal
#import geopandas
import core.helpers as fn
import core.event_extraction as ee

#sampling_time=1
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