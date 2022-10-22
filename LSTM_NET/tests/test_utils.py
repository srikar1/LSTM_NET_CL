import contextlib
import io
import sys
import inspect
import unittest


@contextlib.contextmanager
def nostdout():
    """A context that can be used to surpress all std outputs of a function.

    The code from this method has been copied from (accessed: 08/14/2019):
        https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

    NOTE Our copyright and license does not apply for this function.
    We use this code WITHOUT ANY WARRANTIES.

    Instead, the code in this method is licensed under CC BY-SA 3.0:
        https://creativecommons.org/licenses/by-sa/3.0/

    The code stems from an answer by Alex Martelli:
        https://stackoverflow.com/users/95810/alex-martelli

    The answer has been editted by Nick T:
        https://stackoverflow.com/users/194586/nick-t
    """
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def unittest_verbosity():
    """Return the verbosity setting of the currently running unittest
    program, or 0 if none is running.
    
    The code from this method has been copied from (accessed: 08/14/2019):
        https://stackoverflow.com/questions/13761697/how-to-access-the-unittest-mainverbosity-setting-in-a-unittest-testcase

    NOTE Our copyright and license does not apply for this function.
    We use this code WITHOUT ANY WARRANTIES.

    Instead, the code in this method is licensed under CC BY-SA 3.0:
        https://creativecommons.org/licenses/by-sa/3.0/

    The code stems from an answer by Gareth Rees:
        https://stackoverflow.com/users/68063/gareth-rees
    """
    frame = inspect.currentframe()
    while frame:
        self = frame.f_locals.get('self')
        if isinstance(self, unittest.TestProgram):
            return self.verbosity
        frame = frame.f_back
    return 0

if __name__ == '__main__':
    pass


