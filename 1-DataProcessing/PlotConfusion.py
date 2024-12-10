#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_confusion(x, y, figname='Confusion.pdf', xlabel=None, ylabel=None, title=None, desired_x_order=[], desired_y_order=[], legend_bbox=1.3, sizex=16, sizey=16):
    x_data = np.array(x)
    y_data = np.array(y)
    confusion_fractions = []
    confusion_fractions_rearr = []
    new_indices = []
    x_labels = []
    
    if list(desired_x_order) != []:
        unique_x = np.array(desired_x_order)
        if not np.array_equal(np.sort(np.unique(x_data)), np.sort(unique_x)):
            raise Exception("desired_x_order does not match data labels")
    else:
        unique_x = np.unique(x_data)

    if list(desired_y_order) != []:
        unique_y = np.array(desired_y_order)
        if not np.array_equal(np.sort(np.unique(y_data)), np.sort(unique_y)):
            raise Exception("desired_y_order does not match data labels")
    else:
        unique_y = np.unique(y_data)
    
    for i in unique_y:
        corr_x = [x_data[j] for j in range(len(x_data)) if y_data[j] == i]
        fractions = []
        for j in unique_x:
            fractions.append(len([k for k in corr_x if k == j])/len(corr_x))
        confusion_fractions.append(fractions)
        new_indices.append(fractions.index(max(fractions)))
        x_labels.append(unique_x[fractions.index(max(fractions))])
    out = confusion_fractions
    
    new_indices_no_dup = []
    [new_indices_no_dup.append(i) for i in new_indices if i not in new_indices_no_dup]
    [new_indices_no_dup.append(i) for i in range(len(unique_x)) if i not in new_indices_no_dup]
    
    x_labels_no_dup = []
    [x_labels_no_dup.append(i) for i in x_labels if i not in x_labels_no_dup]
    [x_labels_no_dup.append(i) for i in unique_x if i not in x_labels_no_dup]
    
    confusion_fractions_col_rearr = copy.deepcopy(confusion_fractions)
    for i in range(len(confusion_fractions)):
        for j in range(len(confusion_fractions[0])):
            confusion_fractions_col_rearr[i][j] = confusion_fractions[i][new_indices_no_dup[j]]
    
    confusion_fractions_rearr = []
    col_labels = []
    switched = []
    for i in range(len(confusion_fractions_col_rearr)):
        if new_indices.count(new_indices[i])>1:
            repeat_row_indices = [j for j in range(len(new_indices)) if new_indices[j]==new_indices[i]]
            if new_indices[i] not in switched:
                [confusion_fractions_rearr.append(confusion_fractions_col_rearr[j]) for j in repeat_row_indices]
                [col_labels.append(unique_y[j]) for j in repeat_row_indices]
                switched.append(new_indices[i])
        else:
            confusion_fractions_rearr.append(confusion_fractions_col_rearr[i])
            col_labels.append(np.unique(y)[i])
    
    confusion_fractions_rearr = np.array(confusion_fractions_rearr)
    
    print(x_labels_no_dup)
    xs = []
    ys = []
    for i in range(len(unique_y)):
        ys.append((i+1)*np.ones(len(x_labels_no_dup)))
        xs.append(np.ones(len(x_labels_no_dup))+np.arange(len(x_labels_no_dup)))
    
    plt.figure(figsize=(12,12))
    plt.xlim([0,len(x_labels_no_dup)+1])
    plt.ylim([len(unique_y)+1,0])
    plt.xticks(ticks=np.ones(len(x_labels_no_dup))+np.arange(len(x_labels_no_dup)),labels=x_labels_no_dup,rotation='vertical')
    plt.yticks(ticks=np.ones(len(unique_y))+np.arange(len(unique_y)),labels=col_labels)
    if xlabel != None:
        plt.xlabel(xlabel, fontsize=sizex)
    if ylabel != None:
        plt.ylabel(ylabel, fontsize=sizey)
    if title != None:
        plt.title(title)
    plt.tight_layout()
    plt.scatter(xs,ys,s=150*confusion_fractions_rearr.flatten(),c=confusion_fractions_rearr.flatten(),cmap='Blues',edgecolors='black',linewidth=0.3)
    leg_xs = (len(x_labels_no_dup)+2)*np.ones(5)
    leg_ys = [1,2,3,4,5]
    leg_scatter = plt.scatter(leg_xs,leg_ys,s=150*np.array([0,0.2,0.4,0.6,1]),c=np.array([0,0.2,0.4,0.6,1]),cmap='Blues')
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    handles, labels = leg_scatter.legend_elements(num=5, prop="sizes", alpha=1)
    labels = ['20%','40%','60%','80%','100%']
    plt.legend(handles, labels, bbox_to_anchor = ((width+legend_bbox)/width,0.5), loc="lower left", title="Sizes")
    cbar = plt.colorbar(shrink=0.6)
    cbar.ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
    plt.grid(True)
    ax.set_axisbelow(True)
    ax.set_aspect('equal')
    plt.savefig(figname, bbox_inches='tight')
    
    return out, unique_x, unique_y, x_labels_no_dup, col_labels

def plot_confusion_forced(x, y, figname='Confusion.pdf', xlabel=None, ylabel=None, title=None, forced_x_order=None, forced_y_order=None, legend_bbox=1.3):
    x_data = np.array(x)
    y_data = np.array(y)
    confusion_fractions = []
    confusion_fractions_rearr = []
    new_indices = []
    x_labels = []
    
    unique_x = np.array(forced_x_order)
    unique_y = np.array(forced_y_order)
    
    for i in unique_y:
        print(i)
        corr_x = [x_data[j] for j in range(len(x_data)) if y_data[j] == i]
        fractions = []
        for j in unique_x:
            fractions.append(len([k for k in corr_x if k == j])/len(corr_x))
        confusion_fractions.append(fractions)
    out = confusion_fractions

    confusion_fractions = np.array(confusion_fractions)
    
    print(unique_x)
    xs = []
    ys = []
    for i in range(len(unique_y)):
        ys.append((i+1)*np.ones(len(unique_x)))
        xs.append(np.ones(len(unique_x))+np.arange(len(unique_x)))
    
    plt.figure(figsize=(12,12))
    plt.xlim([0,len(unique_x)+1])
    plt.ylim([len(unique_y)+1,0])
    plt.xticks(ticks=np.ones(len(unique_x))+np.arange(len(unique_x)),labels=unique_x,rotation='vertical')
    plt.yticks(ticks=np.ones(len(unique_y))+np.arange(len(unique_y)),labels=unique_y)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if title != None:
        plt.title(title)
    plt.tight_layout()
    plt.scatter(xs,ys,s=150*confusion_fractions.flatten(),c=confusion_fractions.flatten(),cmap='Blues')
    leg_xs = (len(unique_x)+2)*np.ones(5)
    leg_ys = [1,2,3,4,5]
    leg_scatter = plt.scatter(leg_xs,leg_ys,s=150*np.array([0,0.2,0.4,0.6,1]),c=np.array([0,0.2,0.4,0.6,1]),cmap='Blues')
    handles, labels = leg_scatter.legend_elements(num=5, prop="sizes", alpha=1)
    labels = ['20%','40%','60%','80%','100%']
    plt.legend(handles, labels, bbox_to_anchor = (legend_bbox,0.5), loc="right", title="Sizes")
    cbar = plt.colorbar(shrink=0.6)
    cbar.ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
    plt.grid(True)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_aspect('equal')
    plt.savefig(figname, bbox_inches='tight')
    
    return out, unique_x, unique_y

def plot_batchbar(labels, batches, column_order=None, figname='BatchBar.pdf'):
    labels = np.array(labels)
    unique_batches = []
    [unique_batches.append(i) for i in batches if i not in unique_batches]
    unique_batches.sort()
    batch_dists = []
    if column_order != None:
        x = column_order
    else:
        x = np.unique(labels)
    for i in x:
        label_batches = [batches[j] for j in range(len(batches)) if labels[j]==i]
        batch_dist = []
        for j in unique_batches:
            batch_fraction = label_batches.count(j)/len(label_batches)
            batch_dist.append(batch_fraction)
        batch_dists.append(batch_dist)
    
    batch_dists = np.array(batch_dists)
    
    plt.figure(figsize=(12,12))
    
    y_prev = np.zeros(len(batch_dists[:,0]))
    for i in range(len(unique_batches)):
        y = batch_dists[:,i]
        if i == 0:
            plt.bar(np.ones(len(x))+np.arange(len(x)),y)
        else:
            plt.bar(np.ones(len(x))+np.arange(len(x)),y,bottom=y_prev)
        y_prev += y
    
    plt.legend(unique_batches, bbox_to_anchor = (1.25,1), loc='upper right')
    plt.xlabel('Cluster')
    plt.ylabel('Fraction')
    plt.xticks(np.ones(len(x))+np.arange(len(x)), labels=x, rotation='vertical')
    plt.xlim([0,len(x)+1])
    plt.ylim([0,1])
    plt.savefig(figname, bbox_inches='tight')
    ax = plt.gca()
    ax.set_axisbelow(True)