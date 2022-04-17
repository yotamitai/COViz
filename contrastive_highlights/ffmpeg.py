from random import randint

import os
import subprocess
from os.path import join

def create_run_file(videos_dir, file_name):
    """Create the necessary files"""
    f = open(join(videos_dir, file_name), "w+")
    f.write("#!/bin/bash\n")
    return f

def ffmpeg_highlights_seperated(videos_dir, n_HLs, verbose=False):
    run_script_name = "generate_videos.sh"
    f1 = create_run_file(videos_dir, run_script_name)

    for i in range(n_HLs):
        f1.write(f"ffmpeg -i HL_{i}.mp4 "
                 f"-vcodec libx264 {randint(100000,900000)}_HL_{i}_reformatted.mp4\n")
    f1.close()

    make_executable(videos_dir, verbose, run_script_name)

def make_executable(videos_dir, verbose, run_file):
    """make executable"""
    current_dir = os.getcwd()
    os.chdir(videos_dir)
    subprocess.call(["chmod", "+x", run_file])
    """call ffmpeg"""
    subprocess.call(f"./{run_file}") if verbose else \
        subprocess.call(f"./{run_file}", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.chdir(current_dir)
