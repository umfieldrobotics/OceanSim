import numpy as np

class ImagingSonarScenario():
    def __init__(self):
        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0


    def setup_scenario(self, rob, sonar):
        self._rob = rob
        self._sonar = sonar        

        self._running_scenario = True




    def teardown_scenario(self):

        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step

