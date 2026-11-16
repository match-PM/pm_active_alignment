from glob import glob
from setuptools import find_packages, setup

package_name = 'active_alignment_simulation'
setup(
    name=package_name, version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'PyYAML'],
    maintainer='mll', maintainer_email='terei@match.uni-hannover.de',
    description='Active alignment simulation batches', license='Apache-2.0',
    entry_points={'console_scripts': [
        'batch_runner = active_alignment_simulation.batch_runner_node:main',
    ]},
)
