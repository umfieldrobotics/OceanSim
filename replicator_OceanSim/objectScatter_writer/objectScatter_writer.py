import omni.replicator.core as rep
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
import random



## Define randomzing functions

# randomly scatter objects on the ground plane

def create_objects(scatter_surface, assets_root_path, numObj):
    usd_files = rep.utils.get_usd_files(path=assets_root_path, recursive=True)
    selected_usd = random.choices(usd_files, k=numObj) # allow duplicate selection, otherwise using random.sample()
    objects_prims = []
    
    for usd in selected_usd:
        object = rep.create.from_usd(usd, semantics=[('class', 'objects')])
    ## TODO
    ## Use semantics to get group of items in replicators
    with objects_prims:
        rep.modify.pose(scale=0.01)
        rep.randomizer.scatter_2d(surface_prims=scatter_surface,
                                check_for_collisions=True) 

rep.randomizer.register(scatter_objects)


groundPlane = rep.get.prims(semantics=[("class", "groundPlane")])
assets_root_path = '/home/haoyu-ma/Desktop/test_usd'
with rep.trigger.on_frame(max_execs=1):
    rep.randomizer.scatter_objects(groundPlane, assets_root_path, 5)