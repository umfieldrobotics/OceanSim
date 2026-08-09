import os
import sys
import json

def main():
    if len(sys.argv) != 2:
        print("Usage: python register_asset_path.py <path_to_assets>")
        sys.exit(1)

    asset_path = sys.argv[1]

    if not os.path.exists(asset_path):
        print(f"Error: Provided path does not exist: {asset_path}")
        sys.exit(1)

    json_data = {"asset_path": os.path.abspath(asset_path)}
    oceansim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    json_path = os.path.join(oceansim_root, 'isaacsim', 'oceansim', 'utils', 'asset_path.json')

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Asset path registered successfully: {json_data['asset_path']}")

if __name__ == "__main__":
    main()
