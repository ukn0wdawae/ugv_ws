from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ugv_audio'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ugv',
    maintainer_email='ugv@localhost',
    description='3D Microphone Array Acoustic Simulation and Coqui Frog Triangulation for ugv_pro_audio',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'coqui_sound_emitter = ugv_audio.coqui_sound_emitter:main',
            'mic_array_localizer = ugv_audio.mic_array_localizer:main',
        ],
    },
)

