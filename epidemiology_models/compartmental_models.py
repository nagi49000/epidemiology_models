import pandas as pd
from datetime import datetime
from datetime import timedelta
from datetime import timezone


class SIRModel:
    def __init__(self, params, initial_conditions):
        """ params - dict<str, float> - parameters for model
                                        keys = {'beta', 'gamma', 'N'}
            initial_conditions - dict<str, float> - initial condition for model
                                                    keys = {'S', 'I', 'R'}
        """
        self._params = params
        self._initial_conditions = initial_conditions

    def _deriv(self, p, c):
        """ p - dict<str, float> - parameters for model
                                   keys = {'beta', 'gamma', 'N'}
            c - dict<str, float> - current state of model
                                   keys = {'S', 'I', 'R'}

            returns time derivative, dict<str, float>, keys = {'S', 'I', 'R'}
        """
        deriv = {}
        deriv['S'] = - p['beta'] * c['I'] * c['S'] / p['N']
        deriv['R'] = p['gamma'] * c['I']
        deriv['I'] = - deriv['S'] - deriv['R']
        return deriv

    def get_numerical_results(self, n_sample, dt_secs, init_time=None):
        """ n_sample - int - number of samples to generate
            dt_secs - float - time step in seconds
            init_time - datetime.datetime - initial timestamp of sample zero

            returns dataframe
        """
        result_list = []
        x = self._initial_conditions.copy()
        result_list.append(x.copy())
        for i_sample in range(1, n_sample):
            deriv = self._deriv(self._params, x)
            x = {k: (v + deriv[k] * dt_secs) for k, v in x.items()}
            result_list.append(x.copy())
        if init_time is None:
            init_time = datetime.fromtimestamp(0, tz=timezone.utc)
        df = pd.DataFrame(result_list)
        df['timestamp'] = [init_time + timedelta(seconds=x*dt_secs)
                           for x in range(n_sample)]
        df.set_index('timestamp')
        return df
