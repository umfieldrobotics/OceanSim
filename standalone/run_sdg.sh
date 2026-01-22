#!/bin/bash
# Bash script to run UWCam_sdg.py on all YAML configs in a specified config directory

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set base directory - adjust this path as needed for your Linux Isaac Sim installation
BASEDIR="/isaac-sim/"

SCRIPT="${BASEDIR}extsUser/isaacsim.oceansim/standalone/UWCam_sdg_seaclear.py"

# Use first argument as config directory, or default if not provided
if [ -z "$1" ]; then
    CONFIG_DIR="${BASEDIR}extsUser/isaacsim.oceansim/standalone/Seaclear_configs"
else
    # If argument is an absolute path (starts with /), use as is; otherwise, make it relative to BASEDIR
    if [[ "$1" == /* ]]; then
        CONFIG_DIR="$1"
    else
        CONFIG_DIR="${BASEDIR}$1"
    fi
fi

echo "Using config directory: \"$CONFIG_DIR\""

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory '$CONFIG_DIR' does not exist!"
    exit 1
fi

# Get start time in seconds
START_TIME=$(date +%s)

# Count YAML files
yaml_count=$(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" -type f | wc -l)

if [ "$yaml_count" -eq 0 ]; then
    echo "No YAML files found in $CONFIG_DIR"
    exit 1
fi

echo "Found $yaml_count YAML file(s) to process"

# Loop through all YAML files in the config directory
for config_file in "$CONFIG_DIR"/*.yaml; do
    # Check if the glob didn't match any files
    if [ ! -f "$config_file" ]; then
        continue
    fi
    
    echo "Running $config_file ..."
    
    # Use python.sh if it exists, otherwise try python3 or python
    if [ -f "${BASEDIR}python.sh" ]; then
        "${BASEDIR}python.sh" "$SCRIPT" --config "$config_file" --close_on_completion
    elif command -v python3 &> /dev/null; then
        python3 "$SCRIPT" --config "$config_file" --close_on_completion
    elif command -v python &> /dev/null; then
        python "$SCRIPT" --config "$config_file" --close_on_completion
    else
        echo "Error: No Python interpreter found!"
        exit 1
    fi
    
    # Check if the Python script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Failed to execute $config_file"
        exit 1
    fi
done

# Get end time
END_TIME=$(date +%s)

# Calculate elapsed time
ELAPSED=$((END_TIME - START_TIME))

echo "Total elapsed time: $ELAPSED seconds"

# Equivalent of 'pause' in Windows - wait for user input
echo "Press any key to continue..."
read -n 1 -s
