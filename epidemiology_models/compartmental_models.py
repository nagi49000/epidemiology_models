import pandas as pd
from datetime import datetime
from datetime import timedelta
from datetime import timezone


class BaseModel:
    """ Generic base model to implement numerical integration of epidemiology models as reported on
        https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
    """

    def __init__(self, params, initial_conditions):
        """ params - dict<str, float> - parameters for model
            initial_conditions - dict<str, float> - initial condition for model
        """
        self._params = params
        self._initial_conditions = initial_conditions

    def _deriv(self, p, c):
        """ p - dict<str, float> - parameters for model
            c - dict<str, float> - current state of model

            returns time derivative, dict<str, float>
        """
        raise NotImplementedError

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

    def get_R0(self):
        """ return the basic reproduction number of the model """
        raise NotImplementedError


class SIRModel(BaseModel):
    """ SIR model as reported on
        https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

        model:
        S -> I -> R
        where:
        S(t) = number of susceptible in population
        I(t) = number of infected in population
        R(t) = number of recovered in population
        S + I + R = N

        model parameters:
        N - int(count) - number of people in the population
        beta - float(Hz) - inverse of the time between contacts
        gamma - float(Hz) - inverse of the recovery time

        optional model parameters:
        Lambda - float(Hz) - population natural death rate
        mu - float(Hz) - birth rate

        R0 = beta/gamma
    """

    def _deriv(self, p, c):
        """ p - dict<str, float> - parameters for model
                                   keys = {'beta', 'gamma', 'N'}
                                   optional keys = {'Lambda', 'mu'}
            c - dict<str, float> - current state of model
                                   keys = {'S', 'I', 'R'}

            returns time derivative, dict<str, float>, keys = {'S', 'I', 'R'}
        """
        p = p.copy()  # leave calling ref unchanged
        p['Lambda'] = p.get('Lambda', 0.0)
        p['mu'] = p.get('mu', 0.0)
        deriv = {}
        deriv['S'] = (p['Lambda'] - p['mu']) * c['S'] - p['beta'] * c['I'] * c['S'] / p['N']
        deriv['R'] = p['gamma'] * c['I'] - p['mu'] * c['R']
        deriv['I'] = - deriv['S'] - deriv['R']
        return deriv

    def get_R0(self):
        """ return the basic reproduction number of the model """
        p = self._params
        return p['beta']/p['gamma']


class SEIRModel(BaseModel):
    """ SEIR model as reported on
        https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

        model:
        S -> E -> I -> R
        where:
        S(t) = number of susceptible in population
        E(t) = number of exposed in population
        I(t) = number of infected in population
        R(t) = number of recovered in population
        S + E + I + R = N

        model parameters:
        N - int(count) - number of people in the population
        beta - float(Hz) - inverse of the time between contacts
        gamma - float(Hz) - inverse of the recovery time
        lambda - float(Hz) - population birth rate
        mu - float(Hz) - population natural death rate
        a - float(Hz) - inverse of average incubation period; from exponential distribution

        R0 = (a/(mu+a))*(beta/(mu+gamma))
    """

    def _deriv(self, p, c):
        """ p - dict<str, float> - parameters for model
                                   keys = {'beta', 'gamma', 'N', 'mu', 'lambda', 'a'}
            c - dict<str, float> - current state of model
                                   keys = {'S', 'I', 'E', 'R'}

            returns time derivative, dict<str, float>, keys = {'S', 'I', 'E', 'R'}
        """
        deriv = {}
        deriv['S'] = (p['lambda'] - p['mu']) * c['S'] - p['beta'] * c['I'] * c['S'] / p['N']
        deriv['R'] = p['gamma'] * c['I'] - p['mu'] * c['R']
        deriv['I'] = p['a'] * c['E'] - (p['gamma'] + p['mu'])*c['I']
        deriv['E'] = - deriv['I'] - deriv['R'] - deriv['S']
        return deriv

    def get_R0(self):
        """ return the basic reproduction number of the model """
        p = self._params
        return p['a'] * p['beta'] / ((p['mu']+p['a']) * (p['mu']+p['gamma']))


class SISModel(BaseModel):
    """ SIS model as reported on
        https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

        model:
        S <-> I
        where:
        S(t) = number of susceptible in population
        I(t) = number of infected in population
        S + I = N

        model parameters:
        N - int(count) - number of people in the population
        beta - float(Hz) - inverse of the time between contacts
        gamma - float(Hz) - inverse of the recovery time

        R0 = beta/gamma
    """

    def _deriv(self, p, c):
        """ p - dict<str, float> - parameters for model
                                   keys = {'beta', 'gamma', 'N'}
            c - dict<str, float> - current state of model
                                   keys = {'S', 'I', 'R'}

            returns time derivative, dict<str, float>, keys = {'S', 'I', 'R'}
        """
        deriv = {}
        deriv['S'] = p['gamma'] * c['I'] - p['beta'] * c['I'] * c['S'] / p['N']
        deriv['I'] = - deriv['S']
        return deriv

    def get_R0(self):
        """ return the basic reproduction number of the model """
        p = self._params
        return p['beta']/p['gamma']
