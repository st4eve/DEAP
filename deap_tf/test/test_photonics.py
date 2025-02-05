import numpy as np
from deap.photonics import MRRTransferFunction, PWB
from deap.photonics import MRMTransferFunction


def test_MRRTransferFunction():
    mrr = MRRTransferFunction(a=1, r1=0.9, r2=0.9)

    phi = np.linspace(-np.pi, np.pi, 1000)

    Tp = mrr.throughput(phi)
    Td = mrr.dropput(phi)

    summed = np.mean(Tp + Td)
    assert summed == 1

    phi_test = np.linspace(0, np.pi, 1000)
    phi_recov = mrr.phaseFromDropput(mrr.dropput(phi_test))

    err = np.abs(phi_test - phi_recov)
    assert np.all(err < 1e-7)

    Td_range = np.linspace(0, 1, 1000)
    assert not np.any(np.isnan(mrr.phaseFromDropput(Td_range)))


def test_MRMTransferFunction():
    mrm = MRMTransferFunction(a=0.9, r=0.9)

    phi_test = np.linspace(0, np.pi, 1000)
    phi_recov = mrm.phaseFromThroughput(mrm.throughput(phi_test))
    err = np.abs(phi_test - phi_recov)
    assert np.all(err < 1e-7)

    Td_range = np.linspace(0, 1, 1000)
    assert not np.any(np.isnan(mrm.phaseFromThroughput(Td_range)))


def test_Photonics():
    phase = np.deg2rad(np.array([180, 175, 10, 20, 0, -10, -15, 20]))
    inputs = [0, 1, 2, 3, 4, 5, 6, 7]
    gain = 1

    pwb = PWB(phase, gain)
    computed = pwb.step(inputs)

    mrr = MRRTransferFunction()
    expected = gain * \
        (np.sum(inputs * mrr.dropput(phase)) -
            np.sum(inputs * mrr.throughput(phase)))

    assert computed == expected

    gain = 0
    pwb = PWB(phase, gain)
    computed = pwb.step(inputs)
    assert computed == 0
