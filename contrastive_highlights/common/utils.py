import os
import glob
import pickle
from os.path import join

import cv2
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte


class Trace(object):
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.game_score = None
        self.length = 0
        self.states = []

    def update(self, obs, r, done, infos, a, state_id):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states.append(state_id)
        self.length += 1


class State(object):
    def __init__(self, name, obs, state, action_vector, feature_vector, img):
        self.observation = obs
        self.image = img
        self.observed_actions = action_vector
        self.name = name
        self.features = feature_vector
        self.state = state

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def make_clean_dirs(path, no_clean=False, file_type=''):
    try:
        os.makedirs(path)
    except:
        if not no_clean: clean_dir(path, file_type)


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def create_video(frames_dir, output_dir, n_HLs, size, fps):
    make_clean_dirs(output_dir)
    for hl in range(n_HLs):
        hl_str = str(hl) if hl > 9 else "0" + str(hl)
        img_array = []
        file_list = sorted(
            [x for x in glob.glob(frames_dir + "/*.png") if x.split('/')[-1].startswith(hl_str)])
        for f in file_list:
            img = cv2.imread(f)
            img_array.append(img)
        out = cv2.VideoWriter(join(output_dir, f'HL_{hl}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def serialize_states(states):
    return ' '.join([str(state[0]) + '_' + str(state[1]) for state in states])

def unserialize_states(string):
    return [tuple(state.split('_')) for state in string.split()]


def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))

def make_same_length(trajectories, horizon, traces):
    """make all trajectories the same length"""
    for d in trajectories:
        if len(d.a1_states) < horizon:
            """insert to start of video"""
            da_traj_idx = d.a1_states.index(d.da_index)
            for _ in range((horizon // 2) - da_traj_idx - 1):
                d.a1_states.insert(0, d.a1_states[0])
                d.a2_states.insert(0, d.a1_states[0])
            """insert to end of video"""
            while len(d.a1_states) < horizon:
                last_idx = d.a1_states[-1]
                if last_idx < len(traces[d.episode].states) - 1:
                    last_idx += 1
                    d.a1_states.append(last_idx)
                else:
                    d.a1_states.append(last_idx)

        for _ in range(horizon - len(d.a2_states)):
            d.a2_states.append(d.a2_states[-1])
    return trajectories

def save_disagreements(a1_DAs, a2_DAs, output_dir, fps):
    disagreement_frames_dir = join(output_dir, "disagreement_frames")
    video_dir = join(output_dir, "videos")
    make_clean_dirs(video_dir)
    make_clean_dirs(join(video_dir, 'temp'))
    make_clean_dirs(disagreement_frames_dir)
    dir = join(video_dir, 'temp')

    height, width, layers = a1_DAs[0][0].shape
    size = (width, height)
    trajectory_length = len(a1_DAs[0])
    da_idx = trajectory_length // 2
    for da_i in range(len(a1_DAs)):
        for img_i in range(len(a1_DAs[da_i])):
            save_image(disagreement_frames_dir, "a1_DA{}_Frame{}".format(str(da_i), str(img_i)),
                       a1_DAs[da_i][img_i])
            save_image(disagreement_frames_dir, "a2_DA{}_Frame{}".format(str(da_i), str(img_i)),
                       a2_DAs[da_i][img_i])

        """up to disagreement"""
        create_video('together' + str(da_i), disagreement_frames_dir, dir, "a1_DA" + str(da_i), size,
                     da_idx, fps, add_pause=[0, 4])
        """from disagreement"""
        name1, name2 = "a1_DA" + str(da_i), "a2_DA" + str(da_i)
        create_video(name1, disagreement_frames_dir, dir, name1, size,
                     trajectory_length, fps, start=da_idx)
        create_video(name2, disagreement_frames_dir, dir, name2, size,
                     trajectory_length, fps, start=da_idx)
    return video_dir