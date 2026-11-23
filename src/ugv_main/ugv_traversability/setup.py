import os
from glob import glob

from setuptools import setup

package_name = 'ugv_traversability'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ugv',
    maintainer_email='user@example.com',
    description='Terrain traversability estimation (elevation map -> slope/step/roughness cost)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traversability_node = ugv_traversability.traversability_node:main',
            'traversability_snapshot = ugv_traversability.snapshot:main',
        ],
    },
)
