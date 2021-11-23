import os
import subprocess
from os.path import join

def create_run_file(videos_dir, file_name):
    """Create the necessary files"""
    f = open(join(videos_dir, file_name), "w+")
    f.write("#!/bin/bash\n")
    return f

def ffmpeg_highlights(name, videos_dir, n_HLs, fade_out_frame, fade_duration, verbose=False):
    """Creates bash file to merge the HL videos and add fade in fade out effects using ffmpeg"""
    run_script_name = "addFadeAndMerge.sh"
    f1 = create_run_file(videos_dir, run_script_name)

    for i in range(n_HLs):
        f1.write(f"ffmpeg -i HL_{i}.mp4 -filter:v "
                 f"'fade=in:{0}:{fade_duration},fade=out:{fade_out_frame}:{fade_duration}' "
                 f"-c:v libx264 -crf 22 -preset veryfast -c:a copy fadeInOut_HL_{i}.mp4\n")

    f1.write(f"ffmpeg -f concat -safe 0 -i list.txt -c copy ../{name}.mp4")
    f1.close()

    f2 = open(join(videos_dir, "list.txt"), "w+")
    for j in range(n_HLs):
        f2.write(f"file fadeInOut_HL_{j}.mp4\n")
    f2.close()

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



def ffmpeg_contrastive(videos_dir, n_HLs, fade_out_frame, fade_duration, verbose=False):
    """Creates bash file to merge the HL videos and add fade in fade out effects using ffmpeg"""

    run_script_name = "addFadeAndMerge.sh"
    f1 = create_run_file(videos_dir, run_script_name)

    """merge with section before important state"""
    for i in range(n_HLs):
        f1.write(f"ffmpeg -f concat -safe 0 -i temp/together_{i}.txt -c copy temp/merged{i}.mp4\n")

    """fade in/out"""
    for i in range(n_HLs):
        f1.write(f"ffmpeg -i temp/merged{i}.mp4 -filter:v "
                 f"'fade=in:{0}:{fade_duration},fade=out:{fade_out_frame}:{fade_duration}' "
                 f"-c:v libx264 -crf 22 -preset veryfast -c:a copy temp/fadeInOut_vid_{i}.mp4\n")

    """concatenate videos"""
    f1.write(f"ffmpeg -f concat -safe 0 -i temp/final_list.txt -c copy Contrastive.mp4")
    f1.close()

    """create files of videos to concatenate"""
    f2 = open(join(videos_dir, "temp/final_list.txt"), "w+")
    for i in range(n_HLs):
        f2.write(f"file fadeInOut_vid_{i}.mp4\n")
    f2.close()

    """create files of videos to concatenate"""
    for i in range(n_HLs):
        f = open(join(videos_dir, f"temp/together_{i}.txt"), "w+")
        f.write(f"file together_{i}.mp4\n")
        f.write(f"file a1_vid{i}.mp4\n")
        f.close()

    make_executable(videos_dir, verbose, run_script_name)
