import os

OCEANSIM_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
ASSETS_PATH = os.path.join(OCEANSIM_ROOT, 'Assets')

def get_oceansim_assets_path() -> str:
    return os.environ.get('OCEANSIM_ASSET_PATH', DEFAULT_ASSETS_PATH)
