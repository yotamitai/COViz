import os
import subprocess
from os.path import join


def side_by_side_video(videos_dir, n_HLs, fade_out_frame, name, fade_duration=2,
                            verbose=False):
    """Creates bash file to merge the HL videos and add fade in fade out effects using ffmpeg"""

    """Create the necessary files"""
    f1 = open(join(videos_dir, "addSideBySide.sh"), "w+")
    f1.write("#!/bin/bash\n")

    """side by side"""
    for i in range(n_HLs):
        f1.write(f"ffmpeg -n -i temp/a1_DA{i}.mp4 -i temp/a2_DA{i}.mp4 "
                 f"-filter_complex "
                 f"vstack"
                 f" temp/apart{i}.mp4\n")

    """scale"""
    for i in range(n_HLs):
        f1.write(f"ffmpeg -i temp/together{i}.mp4 -vf scale=1600:500,setsar=1:1 temp/togetherscaled{i}.mp4\n")
        f1.write(f"ffmpeg -i temp/apart{i}.mp4 -vf scale=1600:500,setsar=1:1 temp/apartscaled{i}.mp4\n")

    """merge with section before disagreement"""
    for i in range(n_HLs):
        f1.write(f"ffmpeg -f concat -safe 0 -i temp/together{i}.txt -c copy temp/merged{i}.mp4\n")

    """fade in/out"""
    for i in range(n_HLs):
        f1.write(f"ffmpeg -i temp/merged{i}.mp4 -filter:v "
                 f"'fade=in:{0}:{fade_duration},fade=out:{fade_out_frame}:{fade_duration}' "
                 f"-c:v libx264 -crf 22 -preset veryfast -c:a copy temp/fadeInOut_HL_{i}.mp4\n")

    """concatenate videos"""
    f1.write(f"ffmpeg -f concat -safe 0 -i temp/final_list.txt -c copy {name}_DA.mp4")
    f1.close()

    """create files of videos to concatenate"""
    f2 = open(join(videos_dir, "temp/final_list.txt"), "w+")
    for i in range(n_HLs):
        f2.write(f"file fadeInOut_HL_{i}.mp4\n")
    f2.close()

    """create files of videos to concatenate"""
    for i in range(n_HLs):
        f = open(join(videos_dir, f"temp/together{i}.txt"), "w+")
        f.write(f"file togetherscaled{i}.mp4\n")
        f.write(f"file apartscaled{i}.mp4\n")
        f.close()

    """make executable"""
    current_dir = os.getcwd()
    os.chdir(videos_dir)
    subprocess.call(["chmod", "+x", "addSideBySide.sh"])
    """call ffmpeg"""
    if verbose:
        subprocess.call("./addSideBySide.sh")
    else:
        subprocess.call("./addSideBySide.sh", stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
    os.chdir(current_dir)  # return
