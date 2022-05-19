import numpy as np

def pitch_similarity(a, b):
    s = len(a)
    num_pitch = np.sum(a < 128)
    acc = np.sum(np.logical_and(a == b,  a < 128)) / num_pitch
    return acc
def rhythm_similarity(a,b):
    s = len(a)
    num_pitch = np.sum(a >= 128)
    acc = np.sum(np.logical_and(a == b,  a >= 128)) / num_pitch
    return acc

def get_loss_and_accuracy():
    data = np.load("res-validate-irish-sketchnet-stage-1.npy", allow_pickle=True)
    racc = 0.0
    pacc = 0.0
    total = 0
    for i, d in enumerate(data):
        inpaint = d["inpaint"]
        gd = d["gd"]
        o_note = gd
        r_note = inpaint
        for j in range(len(o_note)):
            x = np.concatenate(o_note[j], -1)
            y = np.concatenate(r_note[j], -1)
            total += 1
            pacc += pitch_similarity(x, y)
            racc += rhythm_similarity(x, y)
    print(len(data))
    print(pacc / total, racc / total)
