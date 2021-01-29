from setuptools import setup, find_packages
import sys, os.path

setup(  name='Gym_Conduzia_Drive',
        version='0.0.1',
        url='https://github.com/sixauto/conduziadrive',
        description='Gym Conduzia Drive',
        install_requires=[
          'pybullet',
          'matplotlib',
          'pyvirtualdisplay',
          'stable_baselines3',
          'numpy',
          'pyglet',
          'box2d-py',
          'gym',
      ],
      python_requires='>=3.6',
)
