from epidemiology_models.compartmental_models import SIRModel
from epidemiology_models.compartmental_models import SISModel
from epidemiology_models.compartmental_models import SEIRModel
from epidemiology_models.compartmental_models import BaseModel
from pytest import approx
from pytest import raises
from datetime import datetime
from datetime import timezone
from datetime import timedelta


def test_BaseModel():
    m = BaseModel({}, {})
    with raises(NotImplementedError):
        m._deriv({}, {})
    with raises(NotImplementedError):
        m.get_R0()


def test_SIRModel():
    t0 = datetime.fromtimestamp(0, tz=timezone.utc)
    N = 1.0e7
    params = {'beta': 0.0002, 'gamma': 0.0001, 'N': N}
    x0 = {'S': N-1, 'I': 1, 'R': 0}
    m = SIRModel(params, x0)
    df = m.get_numerical_results(100, 3600)
    assert m.get_R0() == approx(2.0)
    assert len(df) == 100
    assert set(df.columns) == {'I', 'R', 'S', 'timestamp'}
    assert list(df[['I', 'R', 'S', 'timestamp']].iloc[0]) == [approx(1.0), approx(0.0), approx(N-1.0), t0]
    assert list(df[['I', 'R', 'S', 'timestamp']].iloc[-1]) == [approx(20.217516248),
                                                               approx(8185628.201958057),
                                                               approx(1814351.5805256944),
                                                               t0+timedelta(seconds=99*3600)]

    params = {'beta': 0.0002, 'gamma': 0.0001, 'N': N, 'Lambda': 0.00001, 'mu': 0.00001}
    m = SIRModel(params, x0)
    df = m.get_numerical_results(100, 3600)
    assert m.get_R0() == approx(2.0)
    assert len(df) == 100
    assert set(df.columns) == {'I', 'R', 'S', 'timestamp'}
    assert list(df[['I', 'R', 'S', 'timestamp']].iloc[0]) == [approx(1.0), approx(0.0), approx(N-1.0), t0]
    assert list(df[['I', 'R', 'S', 'timestamp']].iloc[-1]) == [approx(914642.9772372266),
                                                               approx(9024169.958855512),
                                                               approx(61187.063907258365),
                                                               t0+timedelta(seconds=99*3600)]


def test_SEIRModel():
    t0 = datetime.fromtimestamp(0, tz=timezone.utc)
    a = 1/(14*24*3600)
    N = 66.44e6
    beta = 1/(3*24*3600)
    gamma = 1/(14*24*3600)
    m = SEIRModel({'beta': beta, 'gamma': gamma, 'N': N, 'mu': 0.0,
                   'lambda': 0.0, 'a': a},
                  {'S': N-1, 'E': 0, 'I': 1, 'R': 0})
    df = m.get_numerical_results(10000, 3600)
    assert m.get_R0() == approx(4.666666666666667)
    assert len(df) == 10000
    assert set(df.columns) == {'S', 'E', 'I', 'R', 'timestamp'}
    assert list(df[['I', 'R', 'S', 'E', 'timestamp']].iloc[0]) == [
        approx(1.0), approx(0.0), approx(N-1.0), approx(0.0), t0]
    assert list(df[['I', 'R', 'S', 'E', 'timestamp']].iloc[-1]) == [approx(2016.5525405483231),
                                                                    approx(65786478.65788153),
                                                                    approx(651069.2547217584),
                                                                    approx(435.5348557787078),
                                                                    t0+timedelta(seconds=9999*3600)]


def test_SISModel():
    t0 = datetime.fromtimestamp(0, tz=timezone.utc)
    N = 1.0e7
    params = {'beta': 0.0002, 'gamma': 0.0001, 'N': N}
    x0 = {'S': N-1, 'I': 1}
    m = SISModel(params, x0)
    df = m.get_numerical_results(100, 3600)
    assert m.get_R0() == approx(2.0)
    assert len(df) == 100
    assert set(df.columns) == {'I', 'S', 'timestamp'}
    assert list(df[['I', 'S', 'timestamp']].iloc[0]) == [approx(1.0), approx(N-1.0), t0]
    assert list(df[['I', 'S', 'timestamp']].iloc[-1]) == [approx(4999999.9981355285),
                                                          approx(5000000.001864466),
                                                          t0+timedelta(seconds=99*3600)]
