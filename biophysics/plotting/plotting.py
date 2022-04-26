#!/usr/bin/env python3

import matplotlib.pyplot as plt

def simple_plot(x, y, marker=None, xlabel=None, ylabel=None, label=None, title=None, color=None, style=None):

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

def make_pdf(pdf_name):

    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    return pdf