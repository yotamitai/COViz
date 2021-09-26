import os
import subprocess
from os.path import join


def merge_and_fade(videos_dir, n_HLs, fade_out_frame, fade_duration, name, verbose=False):
    """Creates bash file to merge the HL videos and add fade in fade out effects using ffmpeg"""

    """Create the necissary files"""
    f1 = open(join(videos_dir, "addFadeAndMerge.sh"), "w+")
    f1.write("#!/bin/bash\n")
    for i in range(n_HLs):
        f1.write(f"ffmpeg -i HL_{i}.mp4 -filter:v "
                 f"'fade=in:{0}:{fade_duration},fade=out:{fade_out_frame}:{fade_duration}' "
                 f"-c:v libx264 -crf 22 -preset veryfast -c:a copy fadeInOut_HL_{i}.mp4\n")
    f1.write(f"ffmpeg -f concat -safe 0 -i list.txt -c copy ../{name}_HL.mp4")
    f1.close()

    f2 = open(join(videos_dir, "list.txt"), "w+")
    for j in range(n_HLs):
        f2.write(f"file fadeInOut_HL_{j}.mp4\n")
    f2.close()

    """make executable"""
    current_dir = os.getcwd()
    os.chdir(videos_dir)
    subprocess.call(["chmod", "+x", "addFadeAndMerge.sh"])
    """call ffmpeg"""
    if verbose:
        subprocess.call("./addFadeAndMerge.sh")
    else:
        subprocess.call("./addFadeAndMerge.sh", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.chdir(current_dir)  # return
