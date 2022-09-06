#!/usr/bin/env python3

import matplotlib.pyplot as plt

def get_fig_ax(num_rows, num_cols, size=(5,5)):

    fig, ax = plt.subplots(num_rows, num_cols, figsize=size)

    return fig, ax

def scatter_plot(x, y, marker=None, xlabel=None, ylabel=None, label=None, title=None, color=None, style=None):

    if style is 'figure':
        plt.style.use('figure')

    fig, ax = plt.subplots(1,1) # Make plot

    if marker is None:
        ax.plot(x,y,'ko',label=label) # Default plot data using black filled circles
    elif (marker is not None) and (color is None):
        ax.plot(x,y,marker,color='k',label=label)
    elif (marker is not None) and (color is not None):
        ax.plot(x,y,marker,color=color,label=label)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax

def contour_plot(ax, data, contours, color, limits, linewidth=2):
 
    ax.contour(data, contours, colors=color, linewidths=linewidth, 
    extent=limits)

    return ax

def make_axis_label_dic(params):

    label_dic = {}
    for k in params['Plot settings'].keys():
        label_dic[k] = params['Plot settings'][k]

    return label_dic

def decorate_axes(fig, ax, label_dic):

    if 'x label' in label_dic.keys():
        ax.set_xlabel(label_dic['x label'])
    if 'y label' in label_dic.keys():
        ax.set_ylabel(label_dic['y label'])
    if 'x limits' in label_dic.keys():
        ax.set_xlim(label_dic['x limits'])
    if 'y limits' in label_dic.keys():
        ax.set_ylim(label_dic['y limits'])
    if 'x ticks' in label_dic.keys():
        ax.set_xticks(label_dic['x ticks'])
    if 'y ticks' in label_dic.keys():
        ax.set_yticks(label_dic['y ticks'])

    return fig, ax

def make_pdf(pdf_name):

    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    return pdf