from setuptools import setup, find_packages
import sys


def main():
    if 'develop' not in sys.argv:
        raise NotImplementedError("Use python setup.py develop.")
    setup(
        name='SCISSORS',
        url='https://github.com/SimTk/scissors',
        description='SCISSORS Calculates Interpolated Shape Signatures over' +
                    'Rapid Overlay of Chemical Structures (ROCS) Space',
        long_description=open('README.md').read(),
        packages=find_packages(),
        install_requires=['numpy']
    )

if __name__ == '__main__':
    main()
