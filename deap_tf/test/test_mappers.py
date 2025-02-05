import numpy as np
from deap.mappers import PWBMapper
from deap.mappers import LaserDiodeArrayMapper
from deap.mappers import ModulatorArrayMapper
from deap.mappers import PWBArrayMapper
from deap.mappers import PhotonicConvolverMapper


def test_PWBMapperTwoSum():
    weights = np.array([1, 1])
    pwb = PWBMapper.build(weights)
    assert pwb.outputGain == 1

    for v in ([1, 1], [2, 2], [-1, 1], [10, 1]):
        computed = pwb.step(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 1e-4)


def test_PWBMapperUnity():
    weights = np.array([1])
    unityNeuron = PWBMapper.build(weights)
    assert unityNeuron.outputGain == 1

    for v in ([1], [2], [-1], [10]):
        computed = unityNeuron.step(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 1e-4)


def test_PWBMapperNegative():
    weights = np.array([-1])
    pwb = PWBMapper.build(weights)
    assert pwb.outputGain == 1

    for v in ([1], [2], [-1], [10]):
        computed = pwb.step(v)
        expected = np.dot(weights, v)
        err = (expected - computed) / expected
        assert np.abs(err < 1e-3)


def test_PWBMapperNull():
    weights = np.array([0])
    unityNeuron = PWBMapper.build(weights)
    assert unityNeuron.outputGain == 1

    for v in ([1], [2], [-1], [10]):
        computed = unityNeuron.step(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 2e-3)


def test_NeuronMapplerMultiple():
    weights = np.array([2, -3, 4, 0, 9])
    pwb = PWBMapper.build(weights)

    for v in ([1, 10, 3, 2, 0], [1, 1, 1, 1, 1], [-7, 0.214, 22, 0.7, 2]):
        computed = pwb.step(v)
        expected = np.dot(weights, v)
        err = (expected - computed) / expected
        assert err < 2e-3


def test_PWBUpdate():
    weights = np.array([1, 1])
    pwb = PWBMapper.build(weights)
    assert pwb.outputGain == 1
    for v in ([1, 1], [2, 2], [-1, 1], [10, 1]):
        computed = pwb.step(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 1e-4)

    weights = np.array([2, 1])
    PWBMapper.updateWeights(pwb, weights)
    assert pwb.outputGain == 2
    for v in ([1, 1], [2, 2], [-1, 1], [10, 1]):
        computed = pwb.step(v)
        expected = np.dot(weights, v)
        assert np.abs(computed - expected < 1e-4)


def test_LaserDiodeMapper():
    inputShape = (3, 3)
    outputShape = (32, 32)
    power = 255
    ld = LaserDiodeArrayMapper.build(inputShape, outputShape, power)

    for row in range(ld.connections.shape[0]):
        for col in range(ld.connections.shape[1]):
            index = ld.connections[row, col]
            assert index[0] == row % ld.shape[0]
            assert index[1] == col % ld.shape[1]

    output = ld.step()
    assert np.all(output == power)


def test_ModulatorArrayMapper():
    intenstiyMatrix = np.array([
        [0.4,  0.3, 0.2, 0.9, 0.9],
        [0.2,  0.3, 0.1, 0.2,  0.4],
        [0.1,  0.9, 0.8, 0.8,  0.4],
        [0.05, 0.2, 0.9, 0.77, 0.43],
        [0.05, 0.1, 0.9, 0.8,  0.9]])

    ma = ModulatorArrayMapper.build(intenstiyMatrix)
    inputs = np.array([
        [0, 1, 2, 4,       9],
        [255, 20, 3, 8,    2],
        [5, 2, 1, 5,       1],
        [101, 203, 2, 11,  1],
        [76, 54, 100, 201, 1]])

    actual = ma.step(inputs)
    expected = intenstiyMatrix * inputs
    err = np.abs(actual - expected)

    assert np.all(err < 1e-9)


def test_ModulatorArrayMapperUpdate():
    intenstiyMatrix = np.array([
        [0.4,  0.3, 0.2, 0.9, 0.9],
        [0.2,  0.3, 0.1, 0.2,  0.4],
        [0.1,  0.9, 0.8, 0.8,  0.4],
        [0.05, 0.2, 0.9, 0.77, 0.43],
        [0.05, 0.1, 0.9, 0.8,  0.9]])

    ma = ModulatorArrayMapper.build(intenstiyMatrix)
    inputs = np.array([
        [0, 1, 2, 4,       9],
        [255, 20, 3, 8,    2],
        [5, 2, 1, 5,       1],
        [101, 203, 2, 11,  1],
        [76, 54, 100, 201, 1]])
    actual = ma.step(inputs)
    expected = intenstiyMatrix * inputs
    err = np.abs(actual - expected)
    assert np.all(err < 1e-9)

    intenstiyMatrix = np.array([
        [0.3,  0.4, 0.1, 0.3, 0.9],
        [0.8,  0.3, 0.2, 0.2,  0.2],
        [0.2,  0.2, 0.8, 0.8,  0.9],
        [0.09, 0.1, 0.2, 0.7, 0.46],
        [0.02, 0.7, 0.5, 0.5,  0.6]])
    ModulatorArrayMapper.updateInputs(ma, intenstiyMatrix)
    actual = ma.step(inputs)
    expected = intenstiyMatrix * inputs
    err = np.abs(actual - expected)
    assert np.all(err < 1e-9)


def test_PWBArrayMapperSumAll():
    kernel = np.ones((3, 3))
    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])
    pa = PWBArrayMapper.build(
            inputs.shape, kernel, 1)
    assert pa.connections.shape == (1, 1, 9, 2)
    convolved = pa.step(inputs)
    expected = inputs.sum()
    assert np.abs(convolved - expected) < 1e3


def test_PWBArrayMapperZero():
    kernel = np.zeros((3, 3))
    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])
    pa = PWBArrayMapper.build(
            inputs.shape, kernel, 1)
    assert pa.connections.shape == (1, 1, 9, 2)
    convolved = pa.step(inputs)
    expected = 0
    assert np.abs(convolved - expected) < 1e3


def test_PWBArrayMapperUnity():
    kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
        ])
    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])
    # Keep dimensonality
    paddedInputs = np.pad(inputs, (1, 1), 'constant')

    pa = PWBArrayMapper.build(
            paddedInputs.shape, kernel, 1)
    assert pa.connections.shape == (3, 3, 9, 2)
    convolved = pa.step(paddedInputs)
    expected = inputs
    assert np.all(np.abs(convolved - expected) < 0.01)

    # Select every other pair
    pa = PWBArrayMapper.build(
            paddedInputs.shape, kernel, 2)
    assert pa.connections.shape == (2, 2, 9, 2)
    convolved = pa.step(paddedInputs)
    expected = np.array([
        [1, 3],
        [7, 9]])
    expected = expected.reshape((expected.shape[0], expected.shape[1]))

    assert np.all(np.abs(convolved - expected) < 0.01)

    # Select only first element, non perfectly aligning
    # convolution.
    pa = PWBArrayMapper.build(
            paddedInputs.shape, kernel, 3)
    assert pa.connections.shape == (1, 1, 9, 2)
    convolved = pa.step(paddedInputs)
    expected = 1
    assert np.all(np.abs(convolved - expected) < 0.01)


def test_PhotonicConvolverMapper():
    kernel = np.ones((3, 3, 1))
    inputs = np.zeros((3, 3, 1))
    inputs[:, :, 0] = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    pa = PhotonicConvolverMapper.build(
        inputs, kernel, 1)
    convolved = pa.step()
    expected = inputs.sum()
    assert np.abs(convolved - expected) < 1e3


def test_PhotonicConvolverZero():
    kernel = np.zeros((3, 3))
    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])
    pa = PhotonicConvolverMapper.build(
        inputs, kernel, 1)
    convolved = pa.step()
    expected = 0
    assert np.abs(convolved - expected) < 1e3
