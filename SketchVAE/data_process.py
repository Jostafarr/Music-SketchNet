import numpy as np
from loader.dataloader import DataLoader

data_path = [
    "data/irish_train.npy",
    "data/irish_validate.npy",
    "data/irish_test.npy"
]

hold_state = 128
rest_state = 129

# process rhythm and pitch tokens
split_size = 24
new_data = []

def extract_note(x, pad_token = 128):
    d = []
    for i in x:
        if i < 128:
            d.append(i)
    ori_d = len(d)
    d.extend([pad_token] * (len(x) - len(d)))
    return np.array(d), ori_d

def extract_rhythm(x, hold_token = 2, rest_token = 3):
    d = []
    for i in x:
        if i < 128:
             d.append(1)
        elif i == hold_state:
             d.append(hold_token)
        else:
             d.append(rest_token)
    return np.array(d)

def process_rythm_and_pitch(x):
    for i, d in enumerate(x):
        d = np.array(d["notes"])
        ds = np.split(d, list(range(split_size, len(d), split_size)))
        data = []
        for sd in ds:
            if len(sd) != split_size:
                continue
            q, k = extract_note(sd)
            if k == 0:
                continue
            s = extract_rhythm(sd)
            data.append([sd, q, s, k])
        new_data.append(data)
        if i % 1000 == 0:
            print("processed:", i)

    final_data = []
    for d in new_data:
        for dd in d:
            final_data.append(dd)
    print(len(final_data))

    return final_data


def process_data():
    data_path = [
        "data/irish_train.npy",
        "data/irish_validate.npy",
        "data/irish_test.npy"
    ]

    train_x = np.load(data_path[0], allow_pickle=True)
    validate_x = np.load(data_path[1], allow_pickle=True)
    test_x = np.load(data_path[2], allow_pickle=True)

    np.save("data/irish_train_chord_rhythm.npy", process_rythm_and_pitch(train_x))
    np.save("data/irish_validate_chord_rhythm.npy", process_rythm_and_pitch(validate_x))
    np.save("data/irish_test_chord_rhythm.npy", process_rythm_and_pitch(test_x))
