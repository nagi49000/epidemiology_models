from epidemiology_models.compartmental_models import SIRModel
from pytest import approx
from datetime import datetime
from datetime import timezone
from datetime import timedelta


def test_SIRModel():
    t0 = datetime.fromtimestamp(0, tz=timezone.utc)
    N = 1.0e7
    params = {'beta': 0.0002, 'gamma': 0.0001, 'N': N}
    x0 = {'S': N-1, 'I': 1, 'R': 0}
    m = SIRModel(params, x0)
    df = m.get_numerical_results(100, 3600)
    assert len(df) == 100
    assert set(df.columns) == {'I', 'R', 'S', 'timestamp'}
    assert list(df.iloc[0]) == [approx(1.0), approx(0.0), approx(N-1.0), t0]
    assert list(df.iloc[-1]) == [approx(20.217516248),
                                 approx(8185628.201958057),
                                 approx(1814351.5805256944),
                                 t0+timedelta(seconds=99*3600)]
