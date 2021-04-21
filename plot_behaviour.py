import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit

# paths
base_dir = '/Users/lauragwilliams/Documents/projects/barakeet/phoneme_seequence/sequence/data'

plt.close()
files = glob.glob('%s/EC*.csv' % (base_dir))
dfs = list()
for f in files:
    df = pd.read_csv(f)
    df = df[np.array([str(s) != 'nan' for s in df['stim_number']])]
    dfs.append(df)
df = pd.concat(dfs)
df = df.reset_index()

# need to fix this -- add another column which reflects the flipped labels
# so the same phoneme is always on the same side
df['word_left'] = [p[0] == w[0] for p, w in zip(df['phoneme_pair'], df['word_end'])]

# for curve fitting
def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

def aggregate(x, y):
    sorted_x = np.sort(df_sub['morph_n'].unique())
    return sorted_x, [y[x==val].mean() for val in sorted_x]

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True)
axs = axs.flatten()
for si, stim_val in enumerate(df['stim_number'].unique()):

    # subset
    df_sub = df.query("stim_number == @stim_val")

    # get stim phoneme pair and word
    pp = df_sub['phoneme_pair'].values[0]
    w1, w2 = np.unique(df_sub['word_end'])

    # loop through words
    for w in [w1, w2]:

        # find the continuum step
        if w[0] == pp[0]:
            col = 'purple'
            if df_sub['word_side'].unique()[0] == 'left':
                flip = False
            else:
                flip = True
        elif w[0] == pp[1]:
            col = 'orange'
            if df_sub['word_side'].unique()[0] == 'right':
                flip = False
            else:
                flip = True
        else:
            print("Cannot find pair...")

        datas = list()
        for morph_n in range(1, 7):
            morph_n = str(morph_n)
            df_sub_morph = df_sub.query("resampled == @morph_n and word_end == @w")
            d = df_sub_morph['slider.response'].values
            d = np.array(d, dtype='float')

            # flip data so that the x-axis means identical acoustic input
            if flip:
                d = np.abs(d - 11)

            # plot data
            axs[si].plot([int(morph_n)]*len(d), d/10., 'o', color='k', ms=0.1)

            if len(d) > 0:
                datas.append(d / 10.)
            else:
                datas.append([0]*20)

        axs[si].violinplot(datas, range(1, 7),
                           showmeans=True, widths=1., showextrema=False,
                           points=50)

        # compute and plot curve
        # x, d_agg = aggregate(df_sub['morph_n'], d)
        # p0 = [np.max(d_agg), np.median(x), 1, np.min(d_agg)]
        # popt, pcov = curve_fit(sigmoid, x, d_agg, p0, method='lm')
        # y = sigmoid(x, *popt)
        # axs[si].plot(x, y, color=col, label=w)

        # labels
        axs[si].set_title('%s-%s' % (w1, w2), fontsize=8)
        axs[si].axis('off')
axs[2].set_xlabel('Morph Step')
axs[2].set_ylabel('Slider Response')
plt.show()
