import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

# %%
def get_summary(f):
    hit_cnt = {}
    total_acc = {}
    hit_acc = {}
    hit_rate = {}

    for summary in summary_iterator(f):
        try:
            # step can be omitted for now
            step = summary.step
            tag = summary.summary.value[0].tag
            value = summary.summary.value[0].simple_value
            if '#hit' in tag:
                if value == 0:
                    # no hit for cache, go to next summary item
                    continue
                else:
                    hit_cnt[step] = value
            if 'cache-acc' in tag:
                hit_acc[step] = value
            elif 'total-acc' in tag:
                total_acc[step] = value
            elif 'hitrate' in tag:
                hit_rate[step] = value
        except:
            print(summary)
    hit_cnt = [hit_cnt[k] for k in sorted(hit_cnt.keys())]
    hit_rate = [hit_rate[k] for k in sorted(hit_rate.keys())]
    hit_acc = [hit_acc[k] for k in sorted(hit_acc.keys())]
    total_acc = [total_acc[k] for k in sorted(total_acc.keys())]
    return hit_cnt, hit_rate, hit_acc, total_acc
# %%
files = [
    'runs/Mar29_13-56-11_uvadm phoneme CTC threshold=50/events.out.tfevents.1680112571.uvadm.799223.0',
    'runs/Mar29_12-35-10_uvadm phoneme CTC threshold=80/events.out.tfevents.1680107710.uvadm.779374.0',
]

smrys = list(map(get_summary, files))
thresholds = [50, 80, 100]

def plot_scatter_and_mean(figure, smrys, thresholds, colors, markers, metric):
    metric_id = {'Hit Cnt': 0, 'Hit Rate': 1, 'Cache Acc': 2, 'Total Acc': 3}[metric]
    ax = figure.add_subplot(4,1,metric_id+1)
    # ax.set_ylim(-.1, 1.1)
    for i, smry in enumerate(smrys):
        x = np.arange(len(smry[metric_id]))
        y = smry[metric_id]
        ax.scatter(x, y, label='threshold=%d' % thresholds[i], color=colors[i], s=8, marker=markers[i])
        ax.axhline(np.mean(smry[metric_id]), linestyle='--', label='avg threshold=%d' % thresholds[i], color=colors[i])
    ax.set_title(metric)
    ax.legend()
# %%
HIT_CNT, HIT_RATE, CACHE_ACC, TOTAL_ACC = 0, 1, 2, 3
colors = ['red', 'blue', 'purple', 'green']
markers = ['o', 'v', '^']
figure = plt.figure(figsize=(8, 20))
# %%
plot_scatter_and_mean(figure, smrys, thresholds, colors, markers, 'Hit Cnt')
# %%
plot_scatter_and_mean(figure, smrys, thresholds, colors, markers, 'Hit Rate')
# %%
plot_scatter_and_mean(figure, smrys, thresholds, colors, markers, 'Cache Acc')
# %%
plot_scatter_and_mean(figure, smrys, thresholds, colors, markers, 'Total Acc')
figure.savefig('fig.pdf')