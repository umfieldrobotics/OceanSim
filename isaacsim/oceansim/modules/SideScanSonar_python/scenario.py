import numpy as np
from omni.replicator.core.scripts.functional import write_np
import omni.replicator.core as rep
class SideScanSonarScenario():
    def __init__(self):
        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0
        self._output_dir = '/home/haoyu/Desktop/viz'
        
    def setup_scenario(self, rob, sonar):
        self._rob = rob
        self._sonar = sonar        
        self._running_scenario = True
        self.backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})


    def teardown_scenario(self):

        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step

        # data = self._sonar.get_data()
        # self.backend.schedule(write_np, data=data['info']['radialDistance'], path='r.npy')
        # self.backend.schedule(write_np, data=data['info']['azimuth'], path='azi.npy')
        # self.backend.schedule(write_np, data=data['info']['elevation'], path='ele.npy')



