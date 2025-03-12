# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard
# from isaacsim.examples.interactive.base_sample import BaseSample
# from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
# from isaacsim.storage.native import get_assets_root_path

# THis can only be used after the scene is loaded 
class keyboard_cmd:
    def __init__(self,
                 base_command: np.array = np.array([0.0, 0.0, 0.0]),
                 input_keyboard_mapping: dict = {
                                        # forward command
                                        "W": [1.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-1.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 1.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -1.0, 0.0],
                                        # rise command
                                        "UP": [0.0, 0.0, 1.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -1.0],
                                        }
                ) -> None:
        self._base_command = base_command

        self._input_keyboard_mapping = input_keyboard_mapping

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    # async def setup_post_load(self) -> None:
    #     self._appwindow = omni.appwindow.get_default_app_window()
    #     self._input = carb.input.acquire_input_interface()
    #     self._keyboard = self._appwindow.get_keyboard()
    #     self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
    #     self._physics_ready = False
    #     self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
    #     await self.get_world().play_async()

    # async def setup_post_reset(self) -> None:
    #     self._physics_ready = False
    #     await self.get_world().play_async()

    # def on_physics_step(self, step_size) -> None:
    #     if self._physics_ready:
    #         self.h1.forward(step_size, self._base_command)
    #     else:
    #         self._physics_ready = True
    #         self.h1.initialize()
    #         self.h1.post_reset()
    #         self.h1.robot.set_joints_default_state(self.h1.default_pos)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""
        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    # def _timeline_timer_callback_fn(self, event) -> None:
    #     if self.h1:
    #         self._physics_ready = False

    # def world_cleanup(self):
    #     world = self.get_world()
    #     self._event_timer_callback = None
    #     if world.physics_callback_exists("physics_step"):
    #         world.remove_physics_callback("physics_step")


    def cleanup(self):
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
