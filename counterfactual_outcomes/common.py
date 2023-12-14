import logging

import os
import glob
import pickle
from os.path import join

import cv2
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte

class Trace(object):
    def __init__(self, idx, k_steps):
        self.obs = []
        self.previous_actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.length = 0
        self.states = []
        self.trace_idx = idx
        self.k_steps = k_steps

    def update(self, obs, r, done, infos, a, state_id):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.previous_actions.append(a)
        self.reward_sum += r
        self.states.append(state_id)
        self.length += 1

    def get_traj_frames(self, idxs):
        frames = []
        for i in idxs:
            frames.append(self.states[i].image)
        return frames


class State(object):
    def __init__(self, id, obs, state, action_vector, img, features):
        self.observation = obs
        self.image = img
        self.observed_actions = action_vector
        self.id = id
        self.features = features
        self.state = state

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


def get_highlight_traj_indxs(highlights):
    traj_indxs = {}
    for hl in highlights:
        traj_indxs[(hl.id[0], hl.id[1])] = [x.id[1] for x in hl.states if
                                            x.id[1] <= hl.traj_end_state]
    return traj_indxs


def save_frames(trajectories_dict, path, contra_rel_idxs=False):
    make_clean_dirs(path)
    for i, hl in enumerate(trajectories_dict):
        for j, f in enumerate(trajectories_dict[hl]):
            vid_num = str(i) if i > 9 else "0" + str(i)
            frame_num = str(j) if j > 9 else "0" + str(j)
            img_name = f"{vid_num}_{frame_num}"
            if contra_rel_idxs and  j == contra_rel_idxs[hl]:
                img_name += "_CA"
            save_image(path, img_name, f)


def save_highlights(img_shape, n_videos, frames_path, videos_path, args):
    """Save Highlight videos"""
    height, width, layers = img_shape
    img_size = (width, height)
    create_highlights_videos(frames_path, videos_path, n_videos, img_size,
                             args.fps, pause=args.pause)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_traces(path):
    return pickle_load(join(path, 'Traces.pkl'))


def save_traces(traces, output_dir, name='Traces.pkl'):
    try:
        os.makedirs(output_dir)
    except:
        pass
    pickle_save(traces, join(output_dir, name))


def make_clean_dirs(path, no_clean=False, file_type=''):
    try:
        os.makedirs(path)
    except:
        if not no_clean: clean_dir(path, file_type)


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def create_highlights_videos(frames_dir, video_dir, n_HLs, size, fps, pause=None):
    make_clean_dirs(video_dir)
    for hl in range(n_HLs):
        hl_str = str(hl) if hl > 9 else "0" + str(hl)
        img_array = []
        file_list = sorted(
            [x for x in glob.glob(frames_dir + "/*.png") if x.split('/')[-1].startswith(hl_str)])
        for i, f in enumerate(file_list):
            img = cv2.imread(f)
            if f.endswith("CA.png") and pause:
                [img_array.append(img) for _ in range(pause)]
            img_array.append(img)

        out = cv2.VideoWriter(join(video_dir, f'HL_{hl}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    return len(img_array)


def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))


def log_msg(msg, verbose=True):
    logging.info(msg)
    if verbose: print(msg)
