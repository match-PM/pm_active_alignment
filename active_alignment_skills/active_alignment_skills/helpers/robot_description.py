"""Parse robot limits without warnings about known rendering/simulator extras."""

from xml.etree import ElementTree

from urdf_parser_py.urdf import URDF


def parse_robot_description(description):
    root = ElementTree.fromstring(description)
    extensions = {
        'robot': {'gazebo', 'ros2_control'},
        'link': {'gazebo'},
        'joint': {'friction'},
        'collision': {'material'},
    }
    for parent in root.iter():
        ignored = extensions.get(parent.tag, ())
        for child in list(parent):
            if child.tag in ignored:
                parent.remove(child)
    return URDF.from_xml_string(ElementTree.tostring(root, encoding='unicode'))
