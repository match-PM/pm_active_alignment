"""Assembly-scene snapshots, world-pose updates, and restoration via services."""

from copy import deepcopy
import threading
import time

from assembly_manager_interfaces.msg import ObjectScene, RefFrame
from assembly_manager_interfaces.srv import (
    CreateRefFrame, DeleteRefFrame, GetScene, ModifyPoseAbsolut)
from geometry_msgs.msg import Pose
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformException, TransformListener

from active_alignment_skills.helpers.ros_helpers import (
    interruptible_sleep, wait_future, wait_service)


def pose_from_values(position, quaternion):
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError('Pose requires three position and four quaternion values.')
    if not np.all(np.isfinite(list(position) + list(quaternion))):
        raise ValueError('Pose must contain finite values.')
    quaternion = Rotation.from_quat(quaternion).as_quat()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
        float, quaternion)
    return pose


def pose_values(pose):
    return ([pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])


class SceneFrames:
    """Never infer parent-relative frame poses to be world poses."""

    def __init__(self, node):
        self.node = node
        self.lock = threading.Lock()
        self.scene = None
        self.originals = {}
        self.touched = []
        self.subscription = node.create_subscription(
            ObjectScene, '/assembly_manager/scene', self._receive, 10,
            callback_group=node.cb_subscriptions)
        self.clients = {
            'get': node.create_client(GetScene, '/assembly_manager/get_scene',
                                      callback_group=node.cb_reentrant),
            'create': node.create_client(CreateRefFrame, '/assembly_manager/create_ref_frame',
                                         callback_group=node.cb_reentrant),
            'modify': node.create_client(
                ModifyPoseAbsolut, '/assembly_manager/modify_frame_absolut',
                callback_group=node.cb_reentrant),
            'delete': node.create_client(DeleteRefFrame, '/assembly_manager/delete_ref_frame',
                                         callback_group=node.cb_reentrant),
        }
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

    def _receive(self, scene):
        with self.lock:
            self.scene = deepcopy(scene)

    def call(self, key, request, timeout, cancelled=lambda: False):
        client = self.clients[key]
        wait_service(client, timeout, cancelled)
        response = wait_future(client.call_async(request), timeout, cancelled)
        if response is None or not response.success:
            raise RuntimeError(f'Assembly-manager {key} request failed.')
        return response

    def prepare(self, config, cancelled):
        self.originals, self.touched = {}, []
        timeout = config['service_timeout']
        scene = self.call('get', GetScene.Request(), timeout, cancelled).scene
        self._receive(scene)
        frames = list(scene.ref_frames_in_scene)
        for obj in scene.objects_in_scene:
            frames.extend(obj.ref_frames)
        current = {frame.frame_name: frame for frame in frames}
        definitions = {frame['frame_name']: frame for frame in config.get('frames', [])}
        managed = set(definitions) | {config['target']['frame_name']}
        self.originals = {name: deepcopy(current.get(name)) for name in managed}
        # Preserve existing scene definitions. YAML definitions only create missing frames.
        for name in definitions:
            if name in current:
                continue
            specification = definitions[name]
            frame = RefFrame(frame_name=name, parent_frame=specification['parent_frame'])
            frame.pose = pose_from_values(specification.get('position', [0, 0, 0]),
                                          specification.get('quaternion', [0, 0, 0, 1]))
            request = CreateRefFrame.Request(ref_frame=frame, recalculate_constraints=False)
            # Track before the call: a timed-out service may still have created the frame.
            self.touched.append(name)
            self.call('create', request, timeout, cancelled)
            current[name] = frame
        if config['target']['frame_name'] not in current:
            raise ValueError('Target frame is absent; provide a frames entry to create it.')

    def world_pose(self, world, name, timeout, cancelled=lambda: False):
        deadline = time.monotonic() + timeout
        while True:
            try:
                transform = self.buffer.lookup_transform(world, name, rclpy.time.Time())
                t, q = transform.transform.translation, transform.transform.rotation
                return pose_from_values([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            except TransformException:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'No transform from {world} to {name}.')
                interruptible_sleep(0.02, cancelled)

    def apply_seed(self, config, baseline, seed, cancelled):
        position, quaternion = pose_values(baseline)
        rotation = Rotation.from_quat(quaternion)
        pose = pose_from_values(np.asarray(position) + rotation.apply(seed['translation']),
                                (rotation * Rotation.from_quat(seed['quaternion'])).as_quat())
        if config['world_frame'] != 'world':
            parent = self.world_pose('world', config['world_frame'],
                                     config['service_timeout'], cancelled)
            parent_position, parent_quaternion = pose_values(parent)
            local_position, local_quaternion = pose_values(pose)
            parent_rotation = Rotation.from_quat(parent_quaternion)
            pose = pose_from_values(
                np.asarray(parent_position) + parent_rotation.apply(local_position),
                (parent_rotation * Rotation.from_quat(local_quaternion)).as_quat())
        name = config['target']['frame_name']
        if name not in self.touched:
            self.touched.append(name)
        request = ModifyPoseAbsolut.Request(frame_name=name, pose=pose, modify_orientation=True)
        self.call('modify', request, config['service_timeout'], cancelled)
        self.verify_world_pose(
            'world',
            name,
            pose,
            config['service_timeout'],
            cancelled)
        return pose

    def verify_world_pose(self, world, name, expected, timeout, cancelled):
        deadline = time.monotonic() + timeout
        ep, eq = pose_values(expected)
        while True:
            actual = self.world_pose(
                world, name, max(
                    0.05, deadline - time.monotonic()), cancelled)
            ap, aq = pose_values(actual)
            error = (Rotation.from_quat(eq).inv() * Rotation.from_quat(aq)).magnitude()
            if np.linalg.norm(np.asarray(ep) - ap) < 1e-9 and error < 1e-7:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Updated target pose did not propagate to TF.')
            interruptible_sleep(0.02, cancelled)

    def restore(self, timeout):
        errors = []
        for name in reversed(self.touched):
            try:
                original = self.originals[name]
                if original is None:
                    self.call('delete', DeleteRefFrame.Request(frame_name=name), timeout)
                else:
                    self.call('create', CreateRefFrame.Request(
                        ref_frame=original, recalculate_constraints=False), timeout)
            except Exception as exc:
                errors.append(f'{name}: {exc}')
        self.touched = []
        if errors:
            raise RuntimeError('Could not restore frames: ' + '; '.join(errors))
