"""Ignore simulator extensions while retaining joint limits and URDF diagnostics."""

import pytest
from urdf_parser_py.xml_reflection.core import ParseError

from active_alignment_skills.helpers.robot_description import parse_robot_description


DESCRIPTION = '''<robot name="test">
  <link name="base"><gazebo><material>Gazebo/Grey</material></gazebo></link>
  <link name="stage">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material>
    </collision>
  </link>
  <joint name="stage_joint" type="prismatic">
    <parent link="base"/><child link="stage"/>
    <limit lower="-0.1" upper="0.2" effort="10" velocity="1"/>
    <dynamics damping="0.2" friction="0.3"/>
    <friction value="0.1"/>
  </joint>
  <gazebo reference="stage"><material>Gazebo/Grey</material></gazebo>
  <ros2_control name="test" type="system"/>
</robot>'''


def test_simulator_extensions_are_quiet_and_limits_are_preserved(capsys):
    robot = parse_robot_description(DESCRIPTION)
    joint = robot.joint_map['stage_joint']
    assert joint.limit.lower == -0.1
    assert joint.limit.upper == 0.2
    assert joint.dynamics.friction == 0.3
    assert joint.dynamics.damping == 0.2
    assert set(robot.link_map) == {'base', 'stage'}
    assert robot.link_map['stage'].visual.material.name == 'red'
    assert robot.link_map['stage'].collision.geometry.size == [0.1, 0.1, 0.1]
    assert capsys.readouterr().err == ''


def test_unexpected_urdf_tags_still_produce_diagnostics(capsys):
    parse_robot_description(DESCRIPTION.replace('</robot>', '<unexpected/></robot>'))
    assert 'Unknown tag "unexpected"' in capsys.readouterr().err


def test_invalid_joint_limits_still_fail():
    with pytest.raises(ParseError, match='lower'):
        parse_robot_description(DESCRIPTION.replace('lower="-0.1"', 'lower="invalid"'))
