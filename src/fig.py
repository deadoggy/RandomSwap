import numpy as np
from matplotlib import pyplot as plt 
import json 



def load_data(fn_format, low_k, up_k):
    
    rs_nmi = [[] for i in range(low_k, up_k)]
    rs_ari = [[] for i in range(low_k, up_k)]
    rs_mse = [[] for i in range(low_k, up_k)]
    km_nmi = [[] for i in range(low_k, up_k)]
    km_ari = [[] for i in range(low_k, up_k)]
    km_mse = [[] for i in range(low_k, up_k)]

    for k in range(low_k, up_k):
        
        with open(fn_format%k) as fin:
            lines = fin.readlines()
            for l in lines:
                if l=='\n':
                    continue

                key = l.split(' ')[0]
                val = eval(l.split(' ')[1])

                if 'rs_nmi' in key:
                    rs_nmi[k-low_k].append(val)
                if 'rs_ari' in key:
                    rs_ari[k-low_k].append(val)
                if 'rs_mse' in key:
                    rs_mse[k-low_k].append(val)
                if 'km_nmi' in key:
                    km_nmi[k-low_k].append(val)
                if 'km_ari' in key:
                    km_ari[k-low_k].append(val)
                if 'km_mse' in key:
                    km_mse[k-low_k].append(val)

    return rs_nmi, rs_ari, rs_mse, km_nmi, km_ari, km_mse


low_k = 2
up_k = 15

rs_nmi, rs_ari, rs_mse, km_nmi, km_ari, km_mse = load_data('out/random_20201019_%d.log', low_k, up_k)


def box(y_1, y_2, title):

    Y = []
    pos = []
    color = []
    label = []
    for i in range(len(y_1)):
        Y.append(y_1[i])
        Y.append(y_2[i])
        pos.append(i+1.-0.13)
        pos.append(i+1.+0.13)
        color.append('blue')
        color.append('red')
        label.append('rs, k=%d'%(i+2))
        label.append('km, k=%d'%(i+2))

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(12,6)

    plot = ax1.boxplot(Y, patch_artist=True, positions=pos, widths=0.2, medianprops={'color':'green'}, showfliers=False)
    
    for i, box in enumerate(plot['boxes']):
        box.set_facecolor(color[i])

    ax1.set_ylabel(title, fontsize=15)
    ax1.set_xticklabels(label, rotation=30)
    ax1.grid()

    plt.savefig('out/box_%s.png'%title, bbox_inches='tight')

box(rs_ari, km_ari, 'ari')
box(rs_nmi, km_nmi, 'nmi')
box(rs_mse, km_mse, 'mse')

def curve(y1, y2, title):

    y1m = [np.mean(i) for i in y1]
    y2m = [np.mean(i) for i in y2]

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(12,6)

    ax1.plot([ str(i) for i in range(low_k, up_k)], y1m, color='blue', label='RandomSwap')
    ax1.plot([ str(i) for i in range(low_k, up_k)], y2m, color='red', label='KMeans')

    ax1.set_ylabel(title, fontsize=15)
    ax1.set_xlabel('k', fontsize=15)
    ax1.set_xticklabels([ str(i) for i in range(low_k, up_k)])
    ax1.grid()
    
    ax1.legend(loc='lower left', bbox_to_anchor=(0.6, 0.9),
               ncol=5, mode=None, borderaxespad=0., frameon=True, fontsize=15,facecolor="#ffffff",columnspacing=.5, handlelength=0.7,handletextpad=0.3)


    plt.savefig('out/mean_%s.png'%title, bbox_inches='tight')



curve(rs_ari, km_ari, 'ari')
curve(rs_nmi, km_nmi, 'nmi')
curve(rs_mse, km_mse, 'mse')

with open('out/time.json') as fin:
    time = json.load(fin)

rs_time = time['rs']
km_time = time['km']
curve(rs_time, km_time, 'time(s)')
