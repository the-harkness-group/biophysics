#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def get_fig_ax(num_rows, num_cols, size=(7,5)):
    "Make figure and axis objects for plotting"

    fig, ax = plt.subplots(num_rows, num_cols, figsize=size)

    return fig, ax

def scatter_plot(ax, x, y, marker_settings, errorbar_settings=None, xerr=None, yerr=None):
    "Make a scatter plot for data visualization, with the ability to include data error bars"

    marker = marker_settings['style']
    mfc = marker_settings['face color']
    mec = marker_settings['edge color']
    ms = marker_settings['size']

    if errorbar_settings:
        ebc = errorbar_settings['color']
        ew = errorbar_settings['line width']
        cs = errorbar_settings['cap size']

    if xerr is not None and yerr is not None:
        ax.errorbar(x,y,yerr=yerr,xerr=xerr,fmt=marker,mfc=mfc,mec=mec,markersize=ms,ecolor=ebc,elinewidth=ew,capsize=cs)
    if xerr is not None and yerr is None:
        ax.errorbar(x,y,xerr=None,fmt=marker,mfc=mfc,mec=mec,markersize=ms,ecolor=ebc,elinewidth=ew,capsize=cs)
    if xerr is None and yerr is not None:
        ax.errorbar(x,y,yerr=yerr,fmt=marker,mfc=mfc,mec=mec,markersize=ms,ecolor=ebc,elinewidth=ew,capsize=cs)
    else:
        ax.plot(x,y,marker,mfc=mfc,mec=mec,markersize=ms)

    return ax

def bar_plot(ax, x_data, y_data, plot_settings):
    "Make bar plot for data visualization"

    for i,v in enumerate(x_data):
        ax.bar(x_data[i],y_data[i],width=plot_settings['bar width'],facecolor=plot_settings['Color'][i],edgecolor='k')

    return ax

def contour_plot(ax, data, contours, color, limits, linewidth=1):
    "Make contour plot for data visualization"
 
    ax.contour(data, contours, colors=color, linewidths=linewidth, 
    extent=limits)

    return ax

def horizontal_line(ax, plot_settings):
    "Add a horizontal line to a plot at a specified y value and x range"

    line_y = plot_settings['horizontal line']['y value']
    line_x_min = plot_settings['horizontal line']['x min value']
    line_x_max = plot_settings['horizontal line']['x max value']
    line_style = plot_settings['horizontal line']['line style']
    line_color = plot_settings['horizontal line']['color']
    ax.hlines(line_y,line_x_min,line_x_max,linestyle=line_style,color=line_color)

    return ax

def y_equals_x_line(ax, plot_settings):
    "Add a y = x line to a plot for assessing a linear correlation"

    import numpy as np
    y_lims = plot_settings['y limits']
    y_line = np.linspace(y_lims[0],y_lims[1],100)
    style = plot_settings['style']
    lw = plot_settings['width']
    lc = plot_settings['color']
    ax.plot(y_line,y_line,style,linewidth=lw,color=lc)
    
    return ax

def polynomial(ax, plot_settings):
    "Plot polynomial for assessing data trend"

    style = plot_settings['style']
    lw = plot_settings['width']
    lc = plot_settings['color']
    xlims = plot_settings['x limits']
    import numpy as np
    x = np.linspace(xlims[0],xlims[1],100)

    poly_order = plot_settings['order']
    poly_coeffs = plot_settings['coefficients']
    y = 0
    for i in range(len(poly_coeffs)):
        y += poly_coeffs[i]*x**i
    ax.plot(x,y,style,linewidth=lw,color=lc)

    return ax

def annotate(ax, x, y, plot_settings, annotations):
    "Annotate data points in a plot with text"

    for i, (x_val, y_val) in enumerate(zip(x,y)):
        ax.annotate(annotations[i],xy=(x_val,y_val),xycoords='data',xytext=(x_val,y_val))

    return ax

def make_axis_label_dic(params):
    "Make dictionary containing plot settings such as axis labels and tick positions"

    label_dic = {}
    for k in params['Plot settings'].keys():
        label_dic[k] = params['Plot settings'][k]

    return label_dic

def mark_point(ax, x, y, plot_settings):
    "Place a marker on a data plot to highlight a point of interest"

    marker = plot_settings['mark point']['style']
    mec = plot_settings['mark point']['face color']
    mfc = plot_settings['mark point']['edge color']
    ms = plot_settings['mark point']['size']

    ax.plot(x,y,marker,mec=mec,mfc=mfc,ms=ms)

    return ax

def decorate_axes(ax, label_dic):
    "Set axis decorations including labels, limits, and ticks"

    if 'axis labels' in label_dic.keys():
        ax.set_xlabel(label_dic['axis labels']['x'])
        ax.set_ylabel(label_dic['axis labels']['y'])
    if 'axis limits' in label_dic.keys():
        ax.set_xlim(label_dic['axis limits']['x'])
        ax.set_ylim(label_dic['axis limits']['y'])
    if 'axis ticks' in label_dic.keys():
        ax.set_xticks(label_dic['axis ticks']['x'])
        ax.set_yticks(label_dic['axis ticks']['y'])
    if label_dic['type'] == 'bar':
        if 'axis ticks' in label_dic.keys():
            ax.set_xticks(label_dic['axis ticks']['x']['values'],labels=label_dic['axis ticks']['x']['labels'],rotation=label_dic['axis ticks']['x']['rotation'])
    if 'title' in label_dic.keys():
        ax.set_title(label_dic['title'])
    if 'legend' in label_dic.keys():
        if label_dic['legend']['show'] == True:
            ax.legend(frameon=False)

    return ax

def save_close(fig, plot_settings):
    "Save a figure with tight layout to a PDF and close the figure"

    fig.tight_layout()
    fig.savefig(f"{plot_settings['plot file name']}",format='pdf',bbox_inches='tight')
    plt.close(fig)
  
def make_pdf(pdf_name):
    "Self explanatory"

    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    return pdf

def cm_to_inches(cm):
    "Matplotlib figure sizes are specified in inches, used in case there is a need convert to cm for publication figure sizes"

    return cm/2.54

# Convert hex color to rgb
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

# Convert rgb to decimal color
def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

# Get continuous hex colormap based on number of elements
def get_continuous_cmap(hex_list, num_elements, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=num_elements)

    hex_cmp = [mcolors.to_hex(cmp(a)) for a in range(num_elements)] # Convert rgb map back into HEX colors for indexing at number of points in dataset

    return hex_cmp

# Get continuous rgba olormap based on number of elements
def get_continuous_rgba_cmap(hex_list, num_elements, cm_space, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=num_elements)
    
    rgba = [cmp(x) for x in cm_space]
    
    return rgba

def get_colors(color_map, num_colors):

    cmap = matplotlib.cm.get_cmap(color_map)
    fracs = [(x+1)/num_colors for x in range(num_colors)]
    colors = cmap(fracs)

    return colors