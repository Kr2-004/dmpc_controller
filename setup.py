from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dmpc_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='puzzlebot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	    'dmpc_node = dmpc_controller.dmpc_node:main',
        'dmpc_copy_node = dmpc_controller.dmpc_copy_node:main',
        'dmpc_cbf_node = dmpc_controller.dmpc_cbf_node:main',
        'damn = dmpc_controller.damn:main',
        'dude = dmpc_controller.dude:main',
        'test = dmpc_controller.test:main',
        ],
    },
)
