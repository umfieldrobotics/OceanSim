import os
import json

OCEANSIM_ASSET_PATH=None

def get_oceansim_assets_path() -> str:
    global OCEANSIM_ASSET_PATH
    
    if OCEANSIM_ASSET_PATH is not None:
        return OCEANSIM_ASSET_PATH
    
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'asset_path.json'))

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"'asset_path.json' not found at {json_path}. "
            "Run 'register_asset_path.py <path_to_assets>' to register the asset path."
        )

    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"'asset_path.json' is not valid JSON: {e}")

    if "asset_path" not in json_data:
        raise KeyError(
            f"'asset_path.json' at {json_path} does not contain the required 'asset_path' key."
        )
        
    asset_path = json_data["asset_path"]
    
    if not os.path.isdir(asset_path):
        raise FileNotFoundError(f"The provided asset path does not exist: {asset_path}. "
                                "Run /path/to/oceansim/config/register_asset_path.py /path/to/assets as described in the ReadMe.")

    OCEANSIM_ASSET_PATH = asset_path
    return asset_path

# Run once to validate on import
get_oceansim_assets_path()

if __name__ == "__main__":
    print("OceanSim Assets are configured at", get_oceansim_assets_path())
