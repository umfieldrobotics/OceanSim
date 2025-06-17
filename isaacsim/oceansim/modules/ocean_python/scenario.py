class OceanScenario():
    def __init__(self):

        self._running_scenario = False
        self._time = 0.0
        
    def setup_scenario(self, water):
        self._running_scenario = True
        self.water = water
        
    def teardown_scenario(self):

        
        self._running_scenario = False
        self._time = 0.0
        



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step
        self.water.deform(self._time)


