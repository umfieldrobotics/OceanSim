from pxr import Usd, Sdf
import omni
stage = omni.usd.get_context().get_stage()
defaultPrim = stage.GetDefaultPrim()

def predicate(prim):
    primName = str(prim.GetName())
	return (primName.startswith("SM") or primName.startswith("FoliageType_")) and prim.HasAuthoredReferences()

for prim in stage.Traverse():            
    if predicate(prim):
        print(prim)