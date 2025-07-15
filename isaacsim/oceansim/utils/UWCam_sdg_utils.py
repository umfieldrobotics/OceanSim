# This is the location of the palletjacks in the simready asset library
PALLETJACKS = [
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Scale_A/PalletTruckScale_A01_PR_NVD_01.usd",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Heavy_Duty_A/HeavyDutyPalletTruck_A01_PR_NVD_01.usd",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd",
]


# The warehouse distractors which will be added to the scene and randomized
DISTRACTORS_WAREHOUSE = 2 * [
    "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1262.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1268.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1482.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1683.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_291.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1454.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1513.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_05.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_E_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
]


import carb 
import omni.replicator.core as rep
import omni.timeline
from isaacsim.storage.native import get_assets_root_path


# needed for loading textures correctly
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception(
            "Nucleus server not found, could not access Isaac Sim assets folder"
        )
    return assets_root_path + relative_path


def full_distractors_list(distractor_type="warehouse"):
    """Distractor type allowed are warehouse, additional or None. They load corresponding objects and add
    them to the scene for DR"""
    full_dist_list = []

    for distractor in DISTRACTORS_WAREHOUSE:
        full_dist_list.append(prefix_with_isaac_asset_server(distractor))

    print("Distractors being added to the scene.")

    return full_dist_list



def add_objects():
    rep_obj_list = [
        rep.create.from_usd(
            palletjack_path, semantics=[("class", "palletjack")], count=2
        )
        for palletjack_path in PALLETJACKS
    ]
    objects_group = rep.create.group(rep_obj_list)
    return objects_group


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [
        rep.create.from_usd(distractor_path, count=1)
        for distractor_path in full_distractors
    ]
    distractor_group = rep.create.group(distractors)
    return distractor_group


def capture_pathtracing(duration=0.0, spp=128):
    timeline = omni.timeline.get_timeline_interface()

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/pathtracing/clampSpp", 0)
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the previous render and motion blur  settings
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


def capture_raytracing2(rt_subframes=-1, pause_timeline=True, duration=0.0,):
    carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")    
    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=duration, pause_timeline=pause_timeline)# Update the app until a given simulation duration has passed (simulate the world between captures)



def run_simulation_loop(duration, simulation_app):
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )


