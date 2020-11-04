import numpy as np
from filtering import arr_bandpass_filter
class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)
        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):
        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]
        trials = []
        classes = []
        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop]
                trial = trial.reshape((1, -1))
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c)

            tt = np.concatenate(t, axis=0)
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


def load_cnt_mrk_y(path):
    ds = MotorImageryDataset(dataset=path)
    cnt = np.transpose(ds.raw)
    mrk = ds.events_position.flatten()
    dur = ds.events_duration.flatten()
    y = np.zeros(len(mrk))
    for i in range(len(mrk)):
        ds.events_type[:,i][0]
        if ds.events_type[:,i][0] not in ds.mi_types:
            y[i] = 5
        elif ds.mi_types[ds.events_type[:,i][0]] == 'left':
            y[i] = 0
        elif ds.mi_types[ds.events_type[:,i][0]] == 'right':
            y[i] = 1
        elif ds.mi_types[ds.events_type[:,i][0]] == 'foot':
            y[i] = 2
        elif ds.mi_types[ds.events_type[:,i][0]] == 'tongue':
            y[i] = 3
        else:
            y[i] = 4
    return cnt, mrk[:-1], dur[:-1], y[1:]
#    return cnt, mrk, dur, y
   

def cnt_to_epo(cnt, mrk, dur):
    epo = []
    for i in range(len(mrk)):
        epo.append(np.array(cnt[mrk[i] : mrk[i] + dur[i]]))
    return np.array(epo)

def out_label_remover(x, y):
    new_x = []
    new_y = []
    for i in range(len(y)):
        if y[i] != 4 and y[i] != 5:
            new_x.append(np.array(x[i]))
            new_y.append(int(y[i]))
    return np.array(new_x), np.array(new_y)


def gen_filtered_data():
    data = []
    for i in range(1, 2):
        path = 'data/comp_iva/A0' + str(i) + 'T.npz'
        cnt, mrk, dur, y = load_cnt_mrk_y(path)
        f_cnt = arr_bandpass_filter(cnt, 8, 30, 250, 5)
        epo = cnt_to_epo(f_cnt, mrk, dur)
        epo, y = out_label_remover(epo, y)
        new_epo = []
        for v in epo:
            new_epo.append(v[750:1500,:])
        new_epo = np.array(new_epo)
        data.append({'x': new_epo, 'y': np.squeeze(y.T)})
    np.savez_compressed('data/test.npz', data = data)
        # scipy.io.savemat('data/c_iv_2a_epo/' + str(i) + '.mat', {'x': new_epo, 'y': np.squeeze(y.T)})




gen_filtered_data()