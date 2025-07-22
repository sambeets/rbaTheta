import time
import os
import multiprocessing
import pandas as pd
from core import save_xls
from core import RBA_theta

os.chdir('.')
BASE_DIR = os.getcwd()
path = os.path.join(BASE_DIR, r'input_data/new_8_wind_turbine_data.xlsx')


def main(path):
    """Main calls functions from core folder and pass to the multiprocessing function

    Args:
        path (directory): where input data is stored in xlsx format

    Returns:
        [xlsx files]: save xlsx files with extracted events in the simulation folder  
    """

    def the_test(path):
        """
        Args:
            path: path of the time series data input
        Returns: significant and stationary events
        """

        wind_data = pd.read_excel(path)
        wind_data = wind_data.iloc[:100, 1:]

        [significant_events, stationary_events, tao] = RBA_theta(data=wind_data,
                                                                 nominal=2.5,
                                                                 s=0.01,
                                                                 k=3,
                                                                 fc=0.3,
                                                                 threshold=0.1)

        save_xls(significant_events,
                 f'simulations/test_results/all_events/significant_events_T_0.1.xlsx')
        save_xls(stationary_events,
                 f'simulations/test_results/all_events/stationary_events_T_0.1.xlsx')

    def multi_proc(path):
        """creates a multi processing object from the specified function

        Args:
            path : directory path of the input file
        """
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

    return multi_proc(path)


if __name__ == '__main__':
    main(path)