from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ugv_coqui_behavior'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml')) + glob(os.path.join('config', '*.xml'))),
        (os.path.join('share', package_name, 'config', 'profiles'), glob(os.path.join('config', 'profiles', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='ugv',
    maintainer_email='ugv@localhost',
    description='Autonomous Coqui Frog Exploration, Sound Triangulation, Camera Pointing & Return Home Behavior',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pt_camera_controller = ugv_coqui_behavior.pt_camera_controller:main',
            'coqui_hunter_node = ugv_coqui_behavior.coqui_hunter_node:main',
            'coqui_survey = ugv_coqui_behavior.coqui_survey_node:main',
            'eval_survey = ugv_coqui_behavior.eval_survey:main',
            'survey_viz_aids = ugv_coqui_behavior.survey_viz_aids:main',
            'gt_odom_bridge = ugv_coqui_behavior.gt_odom_bridge:main',
            'wheel_odom_relay = ugv_coqui_behavior.wheel_odom_relay:main',
            'lidar_localizer = ugv_coqui_behavior.lidar_localizer:main',
        ],
    },
)

