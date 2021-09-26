import glob
import json
import os
import shutil
import pickle
from os.path import join
import cv2

from skimage import img_as_ubyte
import imageio


def load_traces(path):
    return pickle_load(join(path, 'Traces.pkl'))



def save_traces(traces, output_dir):
    os.makedirs(output_dir)
    pickle_save(traces, join(output_dir, 'Traces.pkl'))


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def json_save(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)


def create_video(name, frame_dir, video_dir, agent_hl, size, length, fps, start=0,
                 add_pause=None):
    img_array = []
    for i in range(start, length):
        img = cv2.imread(os.path.join(frame_dir, agent_hl + f'_Frame{i}.png'))
        img_array.append(img)

    if add_pause:
        img_array = [img_array[0] for _ in range(add_pause[0])] + img_array
        img_array = img_array + [img_array[-1] for _ in range(add_pause[1])]

    out = cv2.VideoWriter(os.path.join(video_dir, name) + '.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))


def clean_dir(path, file_type='', hard=False):
    if not hard:
        files = glob.glob(path + "/*" + file_type)
        for f in files:
            os.remove(f)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def make_clean_dirs(path, no_clean=False, file_type='', hard=False):
    try:
        os.makedirs(path)
    except:  # if exists
        if not no_clean: clean_dir(path, file_type, hard)


def mark_agent(img, position=None, color=255, thickness=2):
    assert position, 'Error - No position provided for marking agent'
    img2 = img.copy()
    top_left = (position[0], position[1])
    bottom_right = (position[0] + 30, position[1] + 15)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)
    return img2


def add_border(image_src, border_thickness=5, color=[0, 0, 0]):
    bt = border_thickness
    x = cv2.copyMakeBorder(image_src, bt, bt, bt, bt, cv2.BORDER_CONSTANT, value=color)
    cv2.imshow("Image", x)
    return x
