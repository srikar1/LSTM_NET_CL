# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import unittest
import sys
import tempfile
import os
import shutil
import contextlib
import time

from toy_example import train
#from tests. test_utils import nostdout
from tests.test_utils import unittest_verbosity

class TrainTestCase(unittest.TestCase):
    def setUp(self):
        pass # Nothing to setup.

    def test_cl_hnet_setup(self):
        """This method tests whether the CL capabilities of the 3 polynomials
        toy regression remain as reported in the readme of the corresponding
        folder."""
        verbosity_level = unittest_verbosity()
        targets = [0.004187723621726036, 0.002387890825048089,
                   0.006071540527045727]

        # Without timestamp, test would get stuck/fail if someone mistakenly
        # starts the test case twice.
        timestamp = int(time.time() * 1000)
        out_dir = os.path.join(tempfile.gettempdir(),
                               'test_cl_hnet_setup_%d' % timestamp)
        my_argv = ['foo', '--no_plots', '--no_cuda', '--beta=0.005',
                   '--emb_size=2', '--n_iter=4001', '--lr_hyper=1e-2',
                   '--data_random_seed=42', '--out_dir=%s' % out_dir]
        sys.argv = list(my_argv)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        if verbosity_level == 2:
            fmse, _, _ = train.run()
        else:
            #with nostdout():
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    fmse, _, _ = train.run()
        shutil.rmtree(out_dir)

        self.assertEqual(len(fmse), len(targets))
        for i in range(len(fmse)):
            self.assertAlmostEqual(fmse[i], targets[i], places=3)

    def tearDown(self):
        pass # Nothing to clean up.

if __name__ == '__main__':
    unittest.main()


