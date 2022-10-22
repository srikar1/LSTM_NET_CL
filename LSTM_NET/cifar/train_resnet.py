# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from cifar import train_args
from cifar import train

if __name__ == '__main__':  
    #print('config = train_args.parse_cmd_argu')
    config = train_args.parse_cmd_arguments(mode='resnet_cifar')

    train.run(config, experiment='resnet')


