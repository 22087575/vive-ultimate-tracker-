import datetime
import json
import logging
import time
import math
import openvr
import numpy as np
import quaternion as qp
from scipy.spatial.transform import Rotation as R
from win_precise_time import sleep
import foxglove
from foxglove import Channel
from foxglove.websocket import (
    Capability,
    ChannelView,
    Client,
    ServerListener,
)
from foxglove.schemas import (
    Pose, PoseInFrame, Quaternion, Timestamp, Vector3,
    FrameTransforms, FrameTransform, SceneUpdate, SceneEntity,
    CubePrimitive, Duration, Color
)

# Sampling rate
SAMPLING_RATE = 120




def precise_wait(duration):
    now = time.time()
    end = now + duration
    if duration >= 0.001:
        sleep(duration)
    while now < end:
        now = time.time()


class VRSystemManager:
    def __init__(self):
        self.vr_system = None

    def initialize_vr_system(self):
        try:
            openvr.init(openvr.VRApplication_Other)
            self.vr_system = openvr.VRSystem()
            print(f"Starting Capture")
        except Exception as e:
            print(f"Failed to initialize VR system: {e}")
            return False
        return True

    def get_tracker_data(self):
        return self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        
    def find_tracker(self, poses):
        """
        Find the first valid tracker or controller device.
        Returns its index, or None if not found.
        """
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bPoseIsValid:
                device_class = self.vr_system.getTrackedDeviceClass(i)
                if device_class in [
                    openvr.TrackedDeviceClass_GenericTracker,
                    openvr.TrackedDeviceClass_Controller,
                ]:
                    serial_number = self.vr_system.getStringTrackedDeviceProperty(
                        i, openvr.Prop_SerialNumber_String)
                    print(f"Using tracker {i} with serial {serial_number}")
                    return i
        return None

    
    def print_discovered_objects(self):
        """
        Print information about discovered VR devices.
        """
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr_system.getTrackedDeviceClass(device_index)
            if device_class != openvr.TrackedDeviceClass_Invalid:
                serial_number = self.vr_system.getStringTrackedDeviceProperty(
                    device_index, openvr.Prop_SerialNumber_String)
                model_number = self.vr_system.getStringTrackedDeviceProperty(
                    device_index, openvr.Prop_ModelNumber_String)
                print(f"Device {device_index}: {serial_number} ({model_number})")
                return device_index
    def shutdown_vr_system(self):
        if self.vr_system:
            openvr.shutdown()


class DataConverter:
    @staticmethod
    def convert_to_6dof(pose_mat):
        position = np.array([pose_mat[i][3] for i in range(3)])
        rot_matrix = np.array([[pose_mat[i][j] for j in range(3)] for i in range(3)])

        rot_obj = R.from_matrix(rot_matrix)

        quat = rot_obj.as_quat()  # x, y, z, w
        euler = rot_obj.as_euler('zyx', degrees=True)  # roll, pitch, yaw in radians

        return position , euler
    
    @staticmethod
    def convert_to_quaternion(pose_mat):
        r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
        if r_w == 0: r_w = 0.0001
        r_x = (pose_mat[2][1] - pose_mat[1][2]) / (4 * r_w)
        r_z = (pose_mat[0][2] - pose_mat[2][0]) / (4 * r_w)
        r_y = (pose_mat[1][0] - pose_mat[0][1]) / (4 * r_w)

        x = pose_mat[0][3]
        z = pose_mat[1][3]
        y = pose_mat[2][3]

        return [x, y, z], [r_w, r_x, r_y, r_z]


class TrackerDataListener(ServerListener):
    def __init__(self) -> None:
        self.subscribers: dict[int, set[str]] = {}
        self.first_subscriber_message_sent = False

    def has_subscribers(self) -> bool:
        return len(self.subscribers) > 0

    def on_subscribe(self, client: Client, channel: ChannelView) -> None:
        logging.info(f"Client {client} subscribed to channel {channel.topic}")
        self.subscribers.setdefault(client.id, set()).add(channel.topic)

    def on_unsubscribe(self, client: Client, channel: ChannelView) -> None:
        logging.info(f"Client {client} unsubscribed from channel {channel.topic}")
        self.subscribers[client.id].remove(channel.topic)
        if not self.subscribers[client.id]:
            del self.subscribers[client.id]
        if not self.has_subscribers():
            self.first_subscriber_message_sent = False


class FoxGloveManager:
    def __init__(self, vr_manager: VRSystemManager):
        self.vr_manager = vr_manager
        self.listener = TrackerDataListener()
        self.server = foxglove.start_server(
            server_listener=self.listener,
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],
        )
        self.json_chan = Channel(topic="/tracker_data", schema={
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "tracker_id": {"type": "number"},
                "position": {"type": "object",
                    "properties": {"x": {"type": "number"},
                                   "y": {"type": "number"},
                                   "z": {"type": "number"}}},
                "rotation": {"type": "object",
                    "properties": {"roll": {"type": "number"},
                                   "pitch": {"type": "number"},
                                   "yaw": {"type": "number"}}}
            }
        })

        self.tracker_index = None  # we will lock on one tracker
        print("\nServer started at ws://localhost:8765")

    def publish_loop(self):
        try:
            while True:
                poses = self.vr_manager.get_tracker_data()

                # If we haven’t picked a tracker yet, find one
                if self.tracker_index is None:
                    self.tracker_index = self.vr_manager.find_tracker(poses)
                    if self.tracker_index is None:
                        print("No tracker found yet...")
                        time.sleep(1)
                        continue


                if not poses[self.tracker_index].bPoseIsValid:
                    print("Selected tracker lost tracking...")
                    time.sleep(1)
                    continue

                if self.listener.has_subscribers():
                    if not self.listener.first_subscriber_message_sent:
                        print("\nFoxglove Studio connected! Sending tracker data...")
                        self.listener.first_subscriber_message_sent = True

                    current_time = time.time()
                    position, euler = DataConverter.convert_to_6dof(
                        poses[self.tracker_index].mDeviceToAbsoluteTracking
                    )
                    pose = DataConverter.convert_to_quaternion(
                        poses[self.tracker_index].mDeviceToAbsoluteTracking
                    )

                    # JSON tracker message
                    message = {
                        "timestamp": current_time,
                        "tracker_id": self.tracker_index,
                        "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
                        "rotation": {"roll": float(euler[0]), "pitch": float(euler[1]), "yaw": float(euler[2])},
                    }
                    self.json_chan.log(message)

                    # Pose message
                    pose_msg = PoseInFrame(
                        timestamp=Timestamp.from_datetime(datetime.datetime.now()),
                        frame_id="world",
                        pose=Pose(
                            position=Vector3(x=pose[0][0], y=pose[0][1], z=pose[0][2]),
                            orientation=Quaternion(w=pose[1][0], x=pose[1][1], y=pose[1][2], z=pose[1][3]),
                        ),
                    )
                    foxglove.log("/robot/pose", pose_msg)

                    # TF transform
                    foxglove.log(
                        "/tf",
                        FrameTransforms(transforms=[
                            FrameTransform(
                                parent_frame_id="world",
                                child_frame_id="tracker",
                                rotation=Quaternion(w=pose[1][0], x=pose[1][1], y=pose[1][2], z=pose[1][3]),
                                translation=Vector3(x=pose[0][0], y=pose[0][1], z=pose[0][2]),
                            )
                        ])
                    )

                    # Cube marker
                    foxglove.log(
                        "/boxes",
                        SceneUpdate(entities=[
                            SceneEntity(
                                frame_id="tracker",
                                id="tracker_1",
                                timestamp=Timestamp.from_datetime(datetime.datetime.now()),
                                lifetime=Duration.from_secs(1.2345),
                                cubes=[CubePrimitive(
                                    pose=Pose(
                                        position=Vector3(x=0, y=0, z=0),
                                        orientation=Quaternion(x=0, y=0, z=0, w=1.0),
                                    ),
                                    size=Vector3(x=0.1, y=0.1, z=0.1),
                                    color=Color(r=1.0, g=0, b=0, a=1),
                                )],
                            )
                        ])
                    )

                precise_wait(1.0 / SAMPLING_RATE)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.vr_manager.shutdown_vr_system()
            self.server.stop()


def main():
    logging.basicConfig(level=logging.INFO)
    foxglove.set_log_level(logging.DEBUG)

    vr_manager = VRSystemManager()
    if not vr_manager.initialize_vr_system():
        return

    fg_manager = FoxGloveManager(vr_manager)
    fg_manager.publish_loop()


if __name__ == "__main__":
    main()
