from setuptools import find_packages, setup

package_name = 'active_alignment_skills'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mll',
    maintainer_email='terei@match.uni-hannover.de',
    description='ROS2 Active Alignment Skills',
    license='Apache-2.0',
    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'active_alignment_skill_node = active_alignment_skills.active_alignment_skill_node:main',
            'dummy_value_publisher = active_alignment_skills.dummy_value_publisher:main'
        ],
        },
)
