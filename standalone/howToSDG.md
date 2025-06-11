# How to run stereo camera Camera SDG in this standalone

Step 1: Install Isaac Sim and OceanSim
    
Step 2: Locate `python.sh` file in the root directory of Isaac Sim
    
Step 3: Run this line at the Isaac Sim root directory 

```bash
./python.sh path/to/this/standalone/stereo_sdg.py --data_dir /home/data/saving/directory
```

More arguments:

    --headless (if you'd like to have a viewport)
    --height  (height of the pic)
    --width   (width of the pic)
    --distractors  (an example to manage assets imported to the scene)


`Notice: I should make underwater render parameters also as an input argument for the standalone, but now it's not. It's now an input to one of the class called StereoCamWriter`
