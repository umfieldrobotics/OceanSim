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


    def cleanup(self):
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
