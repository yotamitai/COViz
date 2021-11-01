import operator

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray

# from highway_disagreements.get_agent import ACTION_DICT
from PIL import ImageFont, ImageDraw, Image


def get_and_mark_frames(disagreements, traces, agent_position=(0, 0), actions=[None, None], color=0):
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        t = traces[d.episode]
        relative_idx = d.da_index - d.a1_states[0]
        # actions = argmax(d.a1_s_a_values[relative_idx]), argmax(d.a2_s_a_values[relative_idx])
        a1_frames, a2_frames = t.get_frames(d.a1_states, d.a2_states, d.trajectory_index)

        if agent_position:
            mark_a1_trajectory(a1_frames, agent_position, color=color)
            da_index = t.trajectory_length // 2 - 1
            relative_positions = get_relative_position(t, d, da_index, relative_idx)
            mark_a2_trajectory(a1_frames, da_index, relative_positions, agent_position, a2_frames, color=255-color)

            mark_frames(a1_frames, a2_frames, da_index, agent_position, actions)
        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)
    return a1_disagreement_frames, a2_disagreement_frames


def mark_a1_trajectory(a1_frames, agent_position, color=255):
    for i in range(len(a1_frames)):
        a1_frames[i] = mark_agent(a1_frames[i], position=agent_position, color=color)

def mark_a2_trajectory(a1_frames, da_index, relative_positions, ref_position, a2_frames, color=255):
    for i in range(len(relative_positions)):
        a1_frames[da_index + i + 1] = mark_trajectory_step(a1_frames[da_index + i + 1],
                                                           relative_positions[i], ref_position,
                                                           a2_frames[da_index + i + 1], color=color)


def mark_trajectory_step(img, rel_pos, ref_pos, temp_img, color=255, thickness=-1):
    img2 = img.copy()
    add_x, add_y = int(rel_pos[1] * 5), int(rel_pos[0] * 10)
    top_left = (ref_pos[0] + add_y, ref_pos[1] + add_x)
    bottom_right = (ref_pos[0] + 30 + add_y, ref_pos[1] + 15 + add_x)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)
    cv2.rectangle(img2,(ref_pos[0] + add_y +4, ref_pos[1] + add_x+4),
                  (ref_pos[0] + 30 + add_y -4, ref_pos[1] + 15 + add_x-4 ), (43,165,0,255), thickness)
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


def mark_frames(a1_frames, a2_frames, da_index, mark_position, actions=[None, None]):
    if mark_position:
        """mark disagreement state"""
        a1_frames[da_index] = mark_agent(a1_frames[da_index], text='Disagreement',
                                         position=mark_position)
        a2_frames[da_index] = a1_frames[da_index]
        """mark chosen action"""
        a1_frames[da_index + 1] = mark_agent(a1_frames[da_index + 1], action=actions[0],
                                             position=mark_position)
        a2_frames[da_index + 1] = mark_agent(a2_frames[da_index + 1], action=actions[1],
                                             position=mark_position, color=0)


def mark_agent(img, action=None, text=None, position=None, color=255, thickness=2):
    assert position, 'Error - No position provided for marking agent'
    img2 = img.copy()
    top_left = (position[0], position[1])
    bottom_right = (position[0] + 30, position[1] + 15)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)

    """add action text"""
    if (action is not None) or text:
        font = ImageFont.truetype('Roboto-Regular.ttf', 20)
        text = text or f'Chosen action: {ACTION_DICT[action]}'
        image = Image.fromarray(img2, 'RGB')
        draw = ImageDraw.Draw(image)
        draw.text((40, 40), text, (255, 255, 255), font=font)
        img_array = asarray(image)
        return img_array

    return img2
