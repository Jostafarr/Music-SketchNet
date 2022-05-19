import os
import copy
import random
import numpy as np
import pretty_midi as pyd
from loader.dataloader import MIDI_Loader


def process_impressionistic(s_dir, dataset_path):
    if os.path.exists("data/impressionistic_train.npy") and os.path.exists("data/impressionistic_validate.npy") and os.path.exists("data/impressionistic_test.npy"):
        print("impressionistic already processed")
        return 
    ml = MIDI_Loader("Irish", minStep=0.5 / 6)
    ml.load(os.path.join(s_dir, dataset_path))

    s = ml.processed_all()

    for i in range(len(s)):
        s[i]["raw"] = ""

    ratio = [int(len(s) * 0.7), int(len(s) * 0.9)]
    random.shuffle(s)
    train_s = s[:ratio[0]]
    validate_s = s[ratio[0]:ratio[1]]
    test_s = s[ratio[1]:]
    print(len(train_s), len(validate_s), len(test_s))

    np.save("data/impressionistic_train.npy", train_s)
    np.save("data/impressionistic_validate.npy", validate_s)
    np.save("data/impressionistic_test.npy", test_s)