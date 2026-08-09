"""
ROS2 cmd_vel subscriber using Isaac Sim's built-in ROS2 bridge
Subscribes to /cmd_vel and applies forces to the robot
"""

import omni.graph.core as og
from pxr import PhysxSchema
import carb
import numpy as np

class CmdVelController:
    """
    Subscribes to /cmd_vel using Isaac Sim's ROS2 bridge and applies forces
    """
    def __init__(self, robot_prim_path="/World/rob"):
        self._robot_prim_path = robot_prim_path
        self._robot_prim = None
        self._force_api = None
        self._graph_path = None
        self._current_linear_vel = [0.0, 0.0, 0.0]
        self._current_angular_vel = [0.0, 0.0, 0.0]
        self._setup_cmd_vel_graph()

    def _setup_cmd_vel_graph(self):
        """Create OmniGraph for cmd_vel subscription using Isaac Sim ROS2 bridge"""
        try:
            # Enable ROS2 bridge extension
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.ros2.bridge")

            keys = og.Controller.Keys
            self._graph_path = "/CmdVelSubscriberGraph"

            # Delete existing graph if present
            import omni.usd
            if omni.usd.get_context().get_stage().GetPrimAtPath(self._graph_path):
                import omni.kit.commands
                omni.kit.commands.execute("DeletePrims", paths=[self._graph_path])

            # Create simplified graph - just subscribe, we'll apply forces in update()
            (graph, nodes, _, _) = og.Controller.edit(
                {
                    "graph_path": self._graph_path,
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                    ],
                    keys.SET_VALUES: [
                        ("ROS2SubscribeTwist.inputs:topicName", "/cmd_vel"),
                        ("ROS2SubscribeTwist.inputs:queueSize", 10),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "ROS2SubscribeTwist.inputs:execIn"),
                    ],
                }
            )

            # Store references to read velocities
            self._twist_node = nodes[1]  # ROS2SubscribeTwist node

            carb.log_info(f"[CmdVelController] Initialized cmd_vel subscriber for robot at {self._robot_prim_path}")
            carb.log_info(f"[CmdVelController] Listening to /cmd_vel topic")

        except Exception as e:
            carb.log_error(f"[CmdVelController] Error setting up cmd_vel subscriber: {e}")
            import traceback
            traceback.print_exc()

    def update(self, robot_prim):
        """Update robot with current velocity commands"""
        if self._twist_node is None or robot_prim is None:
            return

        try:
            # Read velocities from OmniGraph node
            linear_vel_attr = og.Controller.attribute(f"{self._graph_path}/ROS2SubscribeTwist.outputs:linearVelocity")
            angular_vel_attr = og.Controller.attribute(f"{self._graph_path}/ROS2SubscribeTwist.outputs:angularVelocity")

            linear_vel = og.Controller.get(linear_vel_attr)
            angular_vel = og.Controller.get(angular_vel_attr)

            if linear_vel is not None:
                self._current_linear_vel = linear_vel
            if angular_vel is not None:
                self._current_angular_vel = angular_vel

            # Apply forces if we have valid commands
            if any(v != 0 for v in self._current_linear_vel) or any(v != 0 for v in self._current_angular_vel):
                # Apply linear force (scaled for underwater)
                force_scale = 50.0
                force = np.array([
                    self._current_linear_vel[0] * force_scale,
                    self._current_linear_vel[1] * force_scale,
                    self._current_linear_vel[2] * force_scale
                ])

                # Apply torque
                torque_scale = 10.0
                torque = np.array([
                    self._current_angular_vel[0] * torque_scale,
                    self._current_angular_vel[1] * torque_scale,
                    self._current_angular_vel[2] * torque_scale
                ])

                # Get or create PhysX force API
                if self._force_api is None:
                    self._force_api = PhysxSchema.PhysxForceAPI.Apply(robot_prim)

                # Apply forces
                self._force_api.GetForceAttr().Set(tuple(force))
                self._force_api.GetTorqueAttr().Set(tuple(torque))

        except Exception as e:
            carb.log_warn(f"[CmdVelController] Error in update: {e}")

    def cleanup(self):
        """Clean up the graph"""
        if self._graph_path:
            try:
                import omni.kit.commands
                omni.kit.commands.execute("DeletePrims", paths=[self._graph_path])
                carb.log_info("[CmdVelController] Cleaned up cmd_vel subscriber")
            except:
                pass
