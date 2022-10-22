# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from mnist.train_splitMNIST import run

if __name__ == '__main__':
    run(mode='perm')