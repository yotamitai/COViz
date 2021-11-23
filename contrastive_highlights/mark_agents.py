import operator
from copy import deepcopy

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray

# from highway_disagreements.get_agent import ACTION_DICT
from PIL import ImageFont, ImageDraw, Image


def get_marked_frames(trajectories, config, static_position=None,
                      actions=None, colors=(0, 255)):
    contrastive_highlights_frames = []
    length = (trajectories[0].trajectory_length // 2) - 1
    for traj in trajectories:
        contrastive_highlights_frames.append(
            mark_frames(traj, length, config, static_pos=static_position,
                        actions=actions, colors=colors))
    return contrastive_highlights_frames


def mark_a1_trajectory(a1_frames, agent_position, colors=[0, 255]):
    for i in range(len(a1_frames)):
        a1_frames[i] = mark_agent(a1_frames[i], position=agent_position, colors=colors)


def mark_a2_trajectory(a1_frames, da_index, relative_positions, ref_position, a2_frames,
                       color=255):
    for i in range(len(relative_positions)):
        a1_frames[da_index + i + 1] = mark_trajectory_step(a1_frames[da_index + i + 1],
                                                           relative_positions[i], ref_position,
                                                           a2_frames[da_index + i + 1],
                                                           colors=colors)


def mark_trajectory_step(img, ref_pos, rect_size=[30, 15], rel_pos=None, colors=[0, 255],
                         thickness=-1):
    img2 = img.copy()
    add_x, add_y = int(rel_pos[1] * 5), int(rel_pos[0] * 10)
    o_top_left = (ref_pos[0] + add_y, ref_pos[1] + add_x)
    o_bottom_right = (ref_pos[0] + rect_size[0] + add_y, ref_pos[1] + rect_size[1] + add_x)

    c_top_left = (ref_pos[0] + add_y + 4, ref_pos[1] + add_x + 4)
    c_bottom_right = (ref_pos[0] + 30 + add_y - 4, ref_pos[1] + 15 + add_x - 4)

    cv2.rectangle(img2, o_top_left, o_bottom_right, colors[0], thickness)
    cv2.rectangle(img2, c_top_left, c_bottom_right, colors[1], thickness)

    """for testing"""
    # plt.imshow(img2)
    # plt.show()
    # plt.imshow(temp_img)
    # plt.show()
    return img2


def get_relative_position(trace, trajectory, da_index, relative_idx):
    a1_obs = np.array([trace.positions[x] for x in
                       trajectory.a1_states[da_index + 1:]])
    a2_obs = np.array([x.position for x in
                       trace.a2_trajectories[trajectory.trajectory_index]][relative_idx + 1:])
    if len(a1_obs) != len(a2_obs):
        a2_obs = np.array(list(a2_obs) + [a2_obs[-1] for _ in range(len(a1_obs) - len(a2_obs))])

    mark_rel_cords = np.around(a2_obs - a1_obs, 3)

    # for i, state in enumerate(a1_obs):
    #     rel_x, rel_y = map(operator.sub, a2_obs[i][0][1:3], state[0][1:3])
    #     mark_rel_cords.append([round(rel_x, 3), round(rel_y, 3)])
    return mark_rel_cords



def mark_frames(trajectory, da_index, config,
                static_pos=False, actions=(None, None), colors=(0, 255)):
    frames = deepcopy(trajectory.frames_original)
    a1_pos = [x.features['position'] for x in trajectory.trajectory_original][da_index:]
    a2_pos = [x.features['position'] for x in trajectory.trajectory_contrastive]

    """mark disagreement state"""
    frames[da_index] = add_text_to_img(frames[da_index], 'Important State', coords=config.text_coords)

    """mark position of agents"""
    for i in range(len(a1_pos)):
        pos = [a1_pos[i], a2_pos[i]]
        frames[da_index + i] = mark_agent(frames[da_index + i], box_size=config.box_size,
                                          positions=pos, colors=colors)

    if static_pos:
        pass

    if actions:
        pass
        # ACTION_DICT[action]
        # add_text_to_img(img,text)

    return frames


def mark_agent(img, positions=None, colors=(100,), thickness=2, add_text=None, box_size=(30,30)):
    assert positions, 'Error - No position provided for marking agent'
    img2 = img.copy()
    for i in range(len(positions)):
        p = positions[i]
        top_left = (p[0], p[1])
        bottom_right = (p[0] + box_size[0], p[1] + box_size[1])
        cv2.rectangle(img2, top_left, bottom_right, colors[i], thickness)
    if add_text:
        img2 = add_text_to_img(img2, add_text)
    return img2


def add_text_to_img(img, text, coords=(40,40)):
    """add action text"""
    img2 = img.copy()
    h,_,_ = img2.shape
    font = ImageFont.truetype('Roboto-Regular.ttf', 30)
    image = Image.fromarray(img2, 'RGB')
    draw = ImageDraw.Draw(image)
    draw.text(coords, text, (255, 0, 0), font=font)
    img_array = asarray(image)
    return img_array
