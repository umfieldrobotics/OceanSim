########################################
#### Here goes setup simulation app ####
########################################


from omni.isaac.kit import SimulationApp
import os
import argparse

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument(
    "--headless",
    type=bool,
    default=False,
    help="Launch script headless, default is False",
)
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument(
    "--num_frames", type=int, default=10, help="Number of frames to record"
)
parser.add_argument(
    "--distractors",
    type=str,
    default="warehouse",
    help="Options are 'warehouse' (default), 'additional' or None",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=os.getcwd() + "/_palletjack_data",
    help="Location where data will be output",
)

args, unknown_args = parser.parse_known_args()

# This is the config used to launch simulation.
CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": args.headless,
    "width": args.width,
    "height": args.height,
    "num_frames": args.num_frames,
}

simulation_app = SimulationApp(launch_config=CONFIG)

# load up OceanSim
import isaacsim.core.utils.extensions as extensions_utils
value = extensions_utils.enable_extension(extension_name='isaacsim.oceansim')
if value:
    print("OceanSim loaded successfully")
else:
    print("OceanSim loaded failed")


#########################################################
#### Here goes scene construction and randomnization ####
#########################################################


import carb
import omni
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import get_current_stage, open_stage
import omni.replicator.core as rep
import numpy as np



## This is the path which has the background scene in which objects will be added.
ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"

# Increase subframes if shadows/ghosting appears of moving objects
# See known issues: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator.html#known-issues
rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)


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


## Additional distractors which can be added to the scene
DISTRACTORS_ADDITIONAL = [
    "/Isaac/Environments/Hospital/Props/Pharmacy_Low.usd",
    "/Isaac/Environments/Hospital/Props/SM_BedSideTable_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_BooksSet_26.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleB.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleA.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleC.usd",
    "/Isaac/Environments/Hospital/Props/SM_Cart_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Chair_02a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Chair_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Computer_02b.usd",
    "/Isaac/Environments/Hospital/Props/SM_Desk_04a.usd",
    "/Isaac/Environments/Hospital/Props/SM_DisposalStand_02.usd",
    "/Isaac/Environments/Hospital/Props/SM_FirstAidKit_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_GasCart_01c.usd",
    "/Isaac/Environments/Hospital/Props/SM_Gurney_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_HospitalBed_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Mirror.usd",
    "/Isaac/Environments/Hospital/Props/SM_MopSet_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd",
    "/Isaac/Environments/Hospital/Props/SM_SupplyCabinet_01c.usd",
    "/Isaac/Environments/Hospital/Props/SM_SupplyCart_01e.usd",
    "/Isaac/Environments/Hospital/Props/SM_TrashCan.usd",
    "/Isaac/Environments/Hospital/Props/SM_Washbasin.usd",
    "/Isaac/Environments/Hospital/Props/SM_WheelChair_01a.usd",
    "/Isaac/Environments/Office/Props/SM_WaterCooler.usd",
    "/Isaac/Environments/Office/Props/SM_TV.usd",
    "/Isaac/Environments/Office/Props/SM_TableC.usd",
    "/Isaac/Environments/Office/Props/SM_Recliner.usd",
    "/Isaac/Environments/Office/Props/SM_Personenleitsystem_Red1m.usd",
    "/Isaac/Environments/Office/Props/SM_Lamp02_162.usd",
    "/Isaac/Environments/Office/Props/SM_Lamp02.usd",
    "/Isaac/Environments/Office/Props/SM_HandDryer.usd",
    "/Isaac/Environments/Office/Props/SM_Extinguisher.usd",
]


# The textures which will be randomized for the wall and floor
TEXTURES = [
    "/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_square_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_concrete_aged_with_lines.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stone_painted_grey.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_various.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_carpet_abstract_pattern.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_siding_weathered_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_animalfur_pattern_greys.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_artificialgrass_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_red_stacked.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_fireplace_wall.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_fabric_square_grid.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_gravel_grey_leaves.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_plastic_blue.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stone_red_hatch.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stucco_red_painted.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stucco_smooth_blue.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
]



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

    if distractor_type == "warehouse":
        for distractor in DISTRACTORS_WAREHOUSE:
            full_dist_list.append(prefix_with_isaac_asset_server(distractor))
    elif distractor_type == "additional":
        for distractor in DISTRACTORS_ADDITIONAL:
            full_dist_list.append(prefix_with_isaac_asset_server(distractor))
    else:
        print("No Distractors being added to the current scene for SDG")

    return full_dist_list


def full_textures_list():
    full_tex_list = []
    for texture in TEXTURES:
        full_tex_list.append(prefix_with_isaac_asset_server(texture))

    return full_tex_list


def add_palletjacks():
    rep_obj_list = [
        rep.create.from_usd(
            palletjack_path, semantics=[("class", "palletjack")], count=2
        )
        for palletjack_path in PALLETJACKS
    ]
    rep_palletjack_group = rep.create.group(rep_obj_list)
    return rep_palletjack_group


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [
        rep.create.from_usd(distractor_path, count=1)
        for distractor_path in full_distractors
    ]
    distractor_group = rep.create.group(distractors)
    return distractor_group


# This will handle replicator
def run_orchestrator():

    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def main():
    # Open the environment in a new stage
    print(f"Loading Stage {ENV_URL}")
    open_stage(prefix_with_isaac_asset_server(ENV_URL))
    stage = get_current_stage()

    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App uppdate {i}..")
        simulation_app.update()

    textures = full_textures_list()
    rep_palletjack_group = add_palletjacks()
    rep_distractor_group = add_distractors(distractor_type=args.distractors)

    

    # Here you can adjust the settings of the stereo camera, link to API:
    # https://docs.omniverse.nvidia.com/py/replicator/1.5.1/source/extensions/omni.replicator.core/docs/API.html?highlight=stereo%20camera#omni.replicator.core.create.stereo_camera
    baseline = 0.5  # meters
    stereo = rep.create.stereo_camera(stereo_baseline=baseline)

    # trigger replicator pipeline
    with rep.trigger.on_frame(max_execs=CONFIG["num_frames"]):

        # Move the camera around in the scene, focus on the center of warehouse
        with stereo:
            rep.modify.pose(
                position=rep.distribution.uniform((-9.2, -11.8, 0.4), (7.2, 15.8, 4)),
                look_at=(0, 0, 0),
            )

        # Get the Palletjack body mesh and modify its color
        with rep.get.prims(path_pattern="SteerAxles"):
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

        # Randomize the pose of all the added palletjacks
        with rep_palletjack_group:
            rep.modify.pose(
                position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                scale=rep.distribution.uniform((0.01, 0.01, 0.01), (0.01, 0.01, 0.01)),
            )

        # Modify the pose of all the distractors in the scene
        with rep_distractor_group:
            rep.modify.pose(
                position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                scale=rep.distribution.uniform(1, 1.5),
            )

        # Randomize the lighting of the scene
        with rep.get.prims(path_pattern="RectLight"):
            rep.modify.attribute(
                "color", rep.distribution.uniform((0, 0, 0), (1, 1, 1))
            )
            rep.modify.attribute(
                "intensity", rep.distribution.normal(100000.0, 600000.0)
            )
            rep.modify.visibility(
                rep.distribution.choice(
                    [True, False, False, False, False, False, False]
                )
            )

        # select floor material
        random_mat_floor = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(textures),
            roughness=rep.distribution.uniform(0, 1),
            metallic=rep.distribution.choice([0, 1]),
            emissive_texture=rep.distribution.choice(textures),
            emissive_intensity=rep.distribution.uniform(0, 1000),
        )

        with rep.get.prims(path_pattern="SM_Floor"):
            rep.randomizer.materials(random_mat_floor)

        # select random wall material
        random_mat_wall = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(textures),
            roughness=rep.distribution.uniform(0, 1),
            metallic=rep.distribution.choice([0, 1]),
            emissive_texture=rep.distribution.choice(textures),
            emissive_intensity=rep.distribution.uniform(0, 1000),
        )

        with rep.get.prims(path_pattern="SM_Wall"):
            rep.randomizer.materials(random_mat_wall)

    # Set up the writer
    writer = rep.WriterRegistry.get("StereoCamWriter")
    # output directory of writer
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)

    # use writer for bounding boxes, rgb and segmentation
    writer.initialize(
        output_dir=output_directory,
    )

    # attach camera render products to writer so that data is outputted
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product = rep.create.render_product(stereo, RESOLUTION)
    writer.attach(render_product)

    # run rep pipeline
    run_orchestrator()
    simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
