import numpy as np
import random
from matplotlib import cm, colors, pyplot as plt
from frozendict import frozendict


# The color of each observation
custom_colors = (
    np.array(
        [
            [0, 0, 0], # black
            [200, 200, 200], # gray
            [255, 102, 102], # light red
            [150, 0, 0], # dark red
            [102, 255, 102], # light green 
            [0, 150, 0], # dark green
            [102, 102, 255], # light blue
            [0, 0, 150], # dark blue
            [255, 255, 255], # white
        ]
    )
    / 256
)

scent_colors = (
    np.array(
        [
            [255, 255, 255], # white
            [255, 102, 102], # light red
            [102, 255, 102], # light green 
            [102, 102, 255], # light blue
            [255, 255, 255], # white

        ]
    )
    / 256
)

# observations that correspond to food
n_scents = 3
perc_size = 3

# position where each food should be placed in the room
food_positions = {1: (0, 1), 2: (1, 2), 3: (2, 1)}

# observation corresponding to a certain food flavor/small amount of food

# The index of the white color, needed for unreachable corner squares
null_obs = len(custom_colors - 1)

def add_food(state):
    pos = state['pos']
    room = state['room']  # Room observations (e.g., walls, free space)
    scent = state['scent']  # Scent types (0: none, 1: cherry, 2: green apple, 3: blueberry)
    food = state['food']

    s = np.random.randint(n_scents) + 1
    scent[1, 0] = s
    food[food_positions[s]] = 1
    scent[food_positions[s]] = s

    return state

# def to_hashable(state):
#     return (tuple(tuple(int(e) for e in row) for row in state['room']), state['pos'])

# def to_dict(state):
#     room, pos = state
#     room_ar = np.zeros((len(room), len(room[0])), dtype=np.int64)
#     for i, row in enumerate(room):
#         for j, e in enumerate(row):
#             room_ar[i, j] = e
#     return {'room': room_ar, 'pos': pos}

def get_cur_perception(state):
    pos = state['pos']
    return state['room'][pos], state['scent'][pos], state['food'][pos]

def step(state, action, hashable=False):
    """Changes the state based on the action."""
    # if hashable:
    #     state = to_dict(state)

    pos = state['pos']
    food = state['food']
    scent = state['scent']

    if food[pos] != 0:
        food[pos] = 0
    if scent[pos] != 0:
        scent[pos] = 0

    if pos == (0, 0):
        pos = (1, 0)
    elif pos == (1, 0) and 1 not in state['food']:
        pos = (0, 0)
        add_food(state)
    else:
        r, c = pos
        if action == 0:
            c -= 1
        elif action == 1:
            c += 1
        elif action == 2:
            r -= 1
        elif action == 3:
            r += 1

        if (r, c) in [(0,1), (1, 0), (1, 1), (1, 2), (2, 1)]:
            pos = r, c

    state['pos'] = pos
    
    # if hashable:
    #     state = to_hashable(state)

    return state

def get_action(key): 
    """Coverts keyboard key to action"""
    # 0: left, 1: right, 2: up, 3: down
    key_map = {'up': 2, 'down': 3, 'left': 0, 'right': 1, '0': 0, '1': 1, '2': 2, '3': 3}
    if key in key_map.keys():
        return key_map[key]
    return None

def plot_room(state, old_plot = None):
    """Plots the game based on state and returns the plot info. If old_plot is given plot info, the function just 
        redraws the old plot with the new state
    """
    pos = state['pos']
    room = state['room']
    scent = state['scent']
    food = state['food']
    r, c = pos

    #"()-().----.     .\n    \\\"/ ___ ;___.'  \n ` ^  ^    "
    #"ᘛ⁐̤ᕐᐷ"
    ASCII_mouse = """()-()\n\\"/\n`"""
    cmap = colors.ListedColormap(custom_colors)
    scent_cmap = colors.ListedColormap(scent_colors)

    if not old_plot:
        fig, ax = plt.subplots()
        scent_mat = ax.matshow(scent, cmap=scent_cmap)
        ax.matshow(room, cmap=cmap, alpha=.6)
        mouse_text = ax.text(c, r, ASCII_mouse, va='center', ha='center', color='black', fontsize=16)
        food_texts = []
        for r, row in enumerate(food):
            for c, f in enumerate(row):
                if f == 1:
                    food_texts.append(ax.text(c, r, '*', va='center', ha='center', color='black', fontsize=16))

    else:
        fig, ax, scent_mat, mouse_text, food_texts = old_plot
        scent_mat.set_data(scent)
        mouse_text.remove()
        mouse_text = ax.text(c, r, ASCII_mouse, va='center', ha='center', color='black', fontsize=16)
        for food_text in food_texts:
            food_text.remove()
        food_texts = []
        for r, row in enumerate(food):
            for c, f in enumerate(row):
                if f == 1:
                    food_texts.append(ax.text(c, r, '*', va='center', ha='center', color='black', fontsize=16))
        fig.canvas.draw()

    return fig, ax, scent_mat, mouse_text, food_texts 



def play_game():
    """Runs and displays game"""
    start_room = np.array(
        [[0, 1, null_obs],
        [1, 1, 1],
        [null_obs, 1, null_obs]],
        dtype=np.uint8
    )
    start_scent = np.array(
        [[0, 0, 0],
        [0, 0, 0],
        [4, 0, 0]],
        dtype=np.uint8
    )

    pos = 0, 0
    state = add_food({'room': start_room, 'scent': start_scent, 'food': np.zeros((3, 3), dtype=np.uint8), 'pos': pos})

    plot_info = plot_room(state)

    def update_image(event):
        nonlocal state, plot_info
        a = get_action(event.key)
        if a in range(4):
            state = step(state, a)
            plot_info = plot_room(state, old_plot=plot_info)

    plot_info[0].canvas.mpl_connect('key_press_event', update_image)
    plt.show()

def datagen_structured_perc_room(length=10000):

    start_room = np.array(
        [[0, 1, null_obs],
        [1, 1, 1],
        [null_obs, 1, null_obs]],
        dtype=np.uint8
    )
    start_scent = np.array(
        [[0, 0, 0],
        [0, 0, 0],
        [4, 0, 0]],
        dtype=np.uint8
    )

    pos = 0, 0
    state = add_food({'room': start_room, 'scent': start_scent, 'food': np.zeros((3, 3), dtype=np.uint8), 'pos': pos})

    actions = np.random.randint(0, 4, length, dtype=np.int64)

    x = np.zeros((length, perc_size), dtype=np.uint8)  # perceptions


    x[0] = get_cur_perception(state)
    
    count = 0
    while count < length - 1:
        state = step(state, actions[count])
        x[count + 1] = get_cur_perception(state)
        count += 1
    return actions, x

if __name__ == '__main__':
    play_game()
