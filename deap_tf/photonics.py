import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from deap.helpers import bisect_min


class MRRTransferFunction:
    """
    Computes the transfer function of a microring resonator (MRR).
    """
    def __init__(self, a=1, r1=0.99, r2=0.99):
        self.a = tf.constant(a, dtype=tf.float32)
        self.r1 = tf.constant(r1, dtype=tf.float32)
        self.r2 = tf.constant(r2, dtype=tf.float32)
        self._minDropput = self.dropput(tf.constant(np.pi, dtype=tf.float32))

    def throughput(self, phi):
        """
        Calculates throuput of a MMR
        """
        num = (self.r2 * self.a)**2 - 2 * self.r1 * self.r2 * self.a * tf.cos(phi) + self.r1**2 # noqa
        denom = 1 - 2 * self.r1 * self.r2 * self.a * tf.cos(phi) + (self.r1 * self.r2 * self.a)**2  # noqa

        return tf.constant(num / denom, dtype=tf.float32)

    def dropput(self, phi):
        """
        Calculates dropput of a MMR
        """
        num = (1 - self.r1**2) * (1 - self.r2**2) * self.a
        denom = 1 - 2 * self.r1 * self.r2 * self.a * tf.cos(phi) + (self.r1 * self.r2 * self.a)**2  # noqa

        return tf.constant(num / denom, dtype=tf.float32)

    def phaseFromDropput(self, Td):
        """
        Given a dropout, create a phase.
        """
        
        Td = tf.constant(Td, dtype=tf.float32)

        # Create empty array to store results
        ans = np.empty(Td.shape)

        # For tiny dropput values, set to pi
        lessThanMin = (Td <= self._minDropput).numpy()
        minOrMore = ~lessThanMin
        ans[lessThanMin] = np.pi

        # For remaning, actually try to solve.
        num = ((1 - self.r1**2) * (1 - self.r2**2) * self.a / Td[minOrMore] - 1 - (self.r1 * self.r2 * self.a)**2) # noqa
        denom = -2 * self.r1 * self.r2 * self.a
        ans[minOrMore] = np.arccos(num / denom)

        return tf.convert_to_tensor(ans)


class MRMTransferFunction:
    """
    Computes the transfer function of a microring modulator (MRM).
    """
    def __init__(self, a=0.9, r=0.9):
        self.a = a
        self.r = r
        self._maxThroughput = self.throughput(tf.constant(np.pi))

    def throughput(self, phi):
        I_pass = self.a**2 - 2 * self.r * self.a * tf.cos(phi) + self.r**2
        I_input = 1 - 2 * self.a * self.r * tf.cos(phi) + (self.r * self.a)**2
        return I_pass / I_input

    def phaseFromThroughput(self, Tn):
        Tn = tf.constant(Tn)

        # Create variable to store results
        ans = tf.experimental.numpy.empty_like(Tn)

        # For high throuputs, set to pi
        moreThanMax = Tn >= self._maxThroughput
        maxOrLess = ~moreThanMax
        ans[moreThanMax] = np.pi

        # Now solve the remainng
        cos_phi = Tn[maxOrLess] * (1 + (self.r * self.a)**2) - self.a**2 - self.r**2  # noqa
        ans[maxOrLess] = np.arccos(cos_phi / (-2 * self.r * self.a * (1 - Tn[maxOrLess])))  # noqa

        return ans


class PWB:
    """
    A simple, time-independent model of a photonic weight bank.
    """
    def __init__(self, phaseShifts, outputGain):
        self.phaseShifts = phaseShifts
        self.outputGain = outputGain
        self.inputSize = phaseShifts.size

        mrr = MRRTransferFunction()
        self._throughput = mrr.throughput(self.phaseShifts)
        self._dropput = mrr.dropput(self.phaseShifts)

    def _update(self, newPhaseShifts, newOutputGain):
        assert self.inputSize == newPhaseShifts.size
        self.phaseShifts = newPhaseShifts
        self.outputGain = newOutputGain

        mrr = MRRTransferFunction()
        self._throughput = mrr.throughput(self.phaseShifts)
        self._dropput = mrr.dropput(self.phaseShifts)

    def step(self, intensities):
        intensities = intensities
        if intensities.size != self.inputSize:
            raise AssertionError(
                    "Number of inputs ({}) is not "
                    "equal to  number of weights ({})".format(
                        intensities.size, self.inputSize))

        summedThroughput = tf.tensordot(intensities, self._throughput,1)
        summedDropput = tf.tensordot(intensities, self._dropput,1)
        photodiodeVoltage = summedDropput - summedThroughput

        return self.outputGain * photodiodeVoltage


class PWBArray:
    def __init__(self, inputShape, connections, pwbs, sharedCounts, stride):
        self.inputShape = inputShape
        assert pwbs.shape == connections.shape[:2]
        self.connections = connections
        assert inputShape == sharedCounts.shape
        self.sharedCounts = sharedCounts
        self.pwbs = pwbs
        self._output = np.empty(self.connections.shape[:2])
        self.stride = stride

    def step(self, intenstiyMatrix):
        intenstiyMatrix = np.asarray(intenstiyMatrix)
        if intenstiyMatrix.shape != self.inputShape:
            raise AssertionError(
                "Input shape {} is not "
                "equal to array shape {}".format(
                    intenstiyMatrix.shape,
                    self.inputShape
                ))

        self._output.fill(0)
        for row in range(self.connections.shape[0]):
            for col in range(self.connections.shape[1]):
                conn = self.connections[row, col]
                inputs = intenstiyMatrix[conn[:, 0], conn[:, 1]]
                sharedCount = self.sharedCounts[conn[:, 0], conn[:, 1]]
                self._output[row, col] += \
                    self.pwbs[row, col].step(
                        (inputs / sharedCount).ravel())

        return self._output


class LaserDiodeArray:
    """
    An array of laser diodes.
    """
    def __init__(self, shape, outputShape, connections, power):
        self.shape = shape
        self.power = power
        self.connections = connections
        self._output = np.ones(outputShape) * power

    def step(self):
        return self._output


class ModulatorArray:
    """
    An array of photonic modulators.
    """
    def __init__(self, phaseShifts):
        self.phaseShifts = np.asarray(phaseShifts)
        self.shape = phaseShifts.shape
        self._mrm = MRMTransferFunction()
        self._throughput = self._mrm.throughput(self.phaseShifts)

    def step(self, intensities):
        intensities = np.asarray(intensities)
        if intensities.shape != self.shape:
            raise AssertionError(
                    "Input shape {} is not "
                    "equal to modulator shape {}").format(
                        intensities.shape,
                        self.inputShape
                    )

        return self._throughput * intensities

    def _update(self, newPhaseShifts):
        self.phaseShifts = np.asarray(newPhaseShifts)
        self._throughput = self._mrm.throughput(self.phaseShifts)


class PhotonicConvolver:
    """
    A photonic convolver, made up different components
    """
    def __init__(self, laserDiodeArray, modulatorArray, pwbArray):
        self.laserDiodeArray = laserDiodeArray
        self.modulatorArray = modulatorArray
        self.pwbArray = pwbArray

    def step(self):
        laserOutput = self.laserDiodeArray.step()
        modulatorOutput = self.modulatorArray.step(laserOutput)
        convOutput = self.pwbArray.step(modulatorOutput)
        return convOutput
