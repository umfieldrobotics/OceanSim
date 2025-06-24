class OceanScenario():
    def __init__(self):

        self._running_scenario = False
        self._time = 0.0
        
    def setup_scenario(self):
        self._running_scenario = True
        
    def teardown_scenario(self):

        
        self._running_scenario = False
        self._time = 0.0
        



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step

    

