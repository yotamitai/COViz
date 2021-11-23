import os
import glob
import pickle
from os.path import join

import cv2
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte

from contrastive_highlights.ffmpeg import ffmpeg_contrastive, ffmpeg_highlights


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
    def __init__(self, name, obs, state, action_vector, features, img):
        self.observation = obs
        self.image = img
        self.observed_actions = action_vector
        self.name = name
        self.features = features
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


def create_highlights_videos(frames_dir, video_dir, n_HLs, size, fps, pause=None):
    make_clean_dirs(video_dir)
    for hl in range(n_HLs):
        hl_str = str(hl) if hl > 9 else "0" + str(hl)
        img_array = []
        file_list = sorted(
            [x for x in glob.glob(frames_dir + "/*.png") if x.split('/')[-1].startswith(hl_str)])
        for i,f in enumerate(file_list):
            img = cv2.imread(f)
            if i == (len(file_list) // 2) - 1 and pause:
                [img_array.append(img) for _ in range(pause)]
            img_array.append(img)

        out = cv2.VideoWriter(join(video_dir, f'HL_{hl}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    return len(img_array)


def serialize_states(states):
    return ' '.join([str(state[0]) + '_' + str(state[1]) for state in states])

def unserialize_states(string):
    return [tuple(state.split('_')) for state in string.split()]

def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))

def save_frames(trajectories, path):
    make_clean_dirs(path)
    for i, frames in enumerate(trajectories):
        for j, frame in enumerate(frames):
            vid_num = str(i) if i > 9 else "0" + str(i)
            frame_num = str(j) if j > 9 else "0" + str(j)
            img_name = f"{vid_num}_{frame_num}"
            save_image(path, img_name, frame)


def save_contrastive_videos(frames, output_dir, fps):
    frames_dir = join(output_dir, "frames")
    video_dir = join(output_dir, "videos")
    make_clean_dirs(video_dir)
    temp_dir = join(video_dir, 'temp')
    make_clean_dirs(temp_dir)
    make_clean_dirs(frames_dir)

    height, width, layers = frames[0][0].shape
    size = (width, height)
    cont_idx = (len(frames[0]) // 2) - 1
    for vid_i in range(len(frames)):
        for img_i in range(len(frames[vid_i])):
            save_image(frames_dir, f"vid{vid_i}_Frame{img_i}", frames[vid_i][img_i])
        """up to disagreement"""
        create_video(f'together_{vid_i}', frames_dir, temp_dir, f"vid{vid_i}", size,
                     cont_idx, fps, add_pause=[0, 4])
        """from disagreement"""
        name1, name2 = f"a1_vid{vid_i}", f"a2_vid{vid_i}"
        create_video(name1, frames_dir, temp_dir, name1, size, len(frames[0]), fps, start=cont_idx)
        create_video(name2, frames_dir, temp_dir, name2, size, len(frames[0]), fps, start=cont_idx)

    """generate video"""
    fade_duration = 2
    fade_out_frame = len(frames[0]) - fade_duration + 11  # +11 from pause in save_disagreements
    # side_by_side_video(video_dir, args.n_disagreements, fade_out_frame, name)
    # ffmpeg_contrastive(video_dir, len(a1), fade_out_frame, fade_duration)
    ffmpeg_highlights(video_dir, len(frames), fade_out_frame, fade_duration)


def create_video(name, frame_dir, video_dir, agent_vid, size, length, fps, start=0,
                 add_pause=None):
    img_array = []
    for i in range(start, length):
        img = cv2.imread(os.path.join(frame_dir, agent_vid + f'_Frame{i}.png'))
        img_array.append(img)

    # plt.imshow(img_array[0])
    # plt.show()

    if add_pause:
        img_array = [img_array[0] for _ in range(add_pause[0])] + img_array
        img_array = img_array + [img_array[-1] for _ in range(add_pause[1])]

    out = cv2.VideoWriter(os.path.join(video_dir, name) + '.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()