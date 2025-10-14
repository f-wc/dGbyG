class InputValueError(Exception):
    pass


class NoPkaError(Exception):
    pass

class UnbalanceError(Exception):
    pass

class NoLicenseError(Exception):
    '''No ChemAxon available licenses found'''
    pass