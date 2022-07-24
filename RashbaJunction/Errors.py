class EnergyOutOfRangeError(Exception):
    """
    The erroe that is raised when for given energy can not be defined neither propagating nor evanescent modes
    """

    pass


class InsulatorError(Exception):
    """
    The error that is raised in the case only evanescent modes are present and the scatterring matrix can not be defined properly
    """
