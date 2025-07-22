"""
This is a fast test that takes a time series array and extacts significant and stationary events.
"""
# importing packages and created modules
import time
import os
import pandas as pd
import multiprocessing
import core.model as model
from core.helpers import save_xls


# function to execute the test with one threshold
def the_test(path):
    """
    Args:
        path: path of the time series data input
    Returns: significant and stationary events
    """

    wind_data = pd.read_excel(path)
    wind_data = wind_data.iloc[:100, 1:]

    [significant_events, stationary_events, tao] = model.RBA_theta(data=wind_data,
                                                                   nominal=2.5,
                                                                   s=0.01,
                                                                   k=3,
                                                                   fc=0.3,
                                                                   threshold=0.1)

    save_xls(significant_events,
             f'simulations/test_results/all_events/significant_events_T_0.1.xlsx')
    save_xls(stationary_events,
             f'simulations/test_results/all_events/stationary_events_T_0.1.xlsx')


# function to test with multiple thresholds to select one that fits a particular data-set
def threshold_test(path):
    """
    Args:
        path: path is carried forward from previous function pointing the location of the times-series input array

    Returns: test results with a range of thresholds
    """
    wind_data = pd.read_excel(path)
    wind_data = wind_data.iloc[:, 1:]

    # test with multiple thresholds
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for t in thresholds:
        [significant_events, stationary_events, tao] = model.RBA_theta(data=wind_data,
                                                                       nominal=2.5,
                                                                       s=0.01, k=3, fc=0.3, threshold=t)
        save_xls(significant_events,
                 f'simulations/test_results/multiple_T/significant_T{t}.xlsx')
        save_xls(stationary_events,
                 f'simulations/test_results/multiple_T/stationary_T{t}.xlsx')


if __name__ == '__main__':
    """
    Multi-processing function to improve the processing time
    Choice: the_test() or threshold_test()
    """
    os.chdir('..')
    BASE_DIR = os.getcwd()
    path = os.path.join(BASE_DIR, r'input_data/new_8_wind_turbine_data.xlsx')

    processes = [
        multiprocessing.Process(target=the_test(path)) for i in range(3)
    ]
    start_time = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    delta_time = time.time() - start_time
    print(f'it takes {delta_time:.1f} seconds to run this test')
