"""Start only the optimizer, dummy publisher, and batch runner from this repository."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    skills = Path(get_package_share_directory('active_alignment_skills'))
    simulation = Path(get_package_share_directory('active_alignment_simulation'))

    optimization_file = LaunchConfiguration('optimization_parameter_file')
    simulation_file = LaunchConfiguration('simulation_parameter_file')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    # Starting the nodes only advertises the actions. A batch runs when a client
    # sends a goal; robot and assembly-manager bringup remain external.
    optimizer = Node(
        package='active_alignment_skills',
        executable='active_alignment_skill_node',
        output='screen',
        parameters=[{
            'initial_parameter_file': optimization_file,
            'use_sim_time': use_sim_time,
        }],
    )
    publisher = Node(
        package='active_alignment_skills',
        executable='dummy_value_publisher',
        output='screen',
        parameters=[{
            'simulation_parameter_file': simulation_file,
            'use_sim_time': use_sim_time,
        }],
    )
    batch_runner = Node(
        package='active_alignment_simulation',
        executable='batch_runner',
        output='screen',
        parameters=[{
            'simulation_parameter_file': simulation_file,
            'use_sim_time': use_sim_time,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'optimization_parameter_file',
            default_value=str(skills / 'config' / 'optimization_parameter.yaml')),
        DeclareLaunchArgument(
            'simulation_parameter_file',
            default_value=str(simulation / 'config' / 'simulation_parameter.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        optimizer,
        publisher,
        batch_runner,
    ])
