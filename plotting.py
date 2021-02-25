"""

"""

import matplotlib
import matplotlib as plt
from matplotlib.patches import Ellipse
import numpy as np


"""
 
  ######  ######## ######## ##     ## ########  
 ##    ## ##          ##    ##     ## ##     ## 
 ##       ##          ##    ##     ## ##     ## 
  ######  ######      ##    ##     ## ########  
       ## ##          ##    ##     ## ##        
 ##    ## ##          ##    ##     ## ##        
  ######  ########    ##     #######  ##        
 
"""

def init_matplotlib_params(save_not_show_fig, show_latex_fig):
    fontsize = 9
    linewidth = 1.0
    gridlinewidth = 0.7

    # Global changes
    matplotlib.rcParams.update({
            # Fonts
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize,
            'figure.titlesize': fontsize,
            # Line width
            'lines.linewidth': linewidth,
            'grid.linewidth': gridlinewidth
        })

    # Backend if saving
    if save_not_show_fig:
        matplotlib.use("pgf")

    # Font if saving or ploting in tex mode
    if save_not_show_fig or show_latex_fig:
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",   
            'font.family': 'serif',         # Use serif/main font for text elements
            'text.usetex': True,            # Use inline maths for ticks
            'pgf.rcfonts': False,           # Don't setup fonts from matplotlib rc params
        })
    return

"""
 
  ######  #### ##     ## 
 ##    ##  ##  ###   ### 
 ##        ##  #### #### 
  ######   ##  ## ### ## 
       ##  ##  ##     ## 
 ##    ##  ##  ##     ## 
  ######  #### ##     ## 
 
"""

def plot_state(plotter, state, **kwargs):
    return plotter.scatter(state[0], state[2], **kwargs)

def plot_state_cov(plotter, state_covariance, state, **kwargs):
    state_2d = np.array([state[0], state[2]])
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[2][0]],
                             [state_covariance[0][2], state_covariance[2][2]]])
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    return plotter.add_artist(ellipse)

def plot_measurement(plotter, measurement, **kwargs):
    ret = None
    if measurement != None:
        ret = plotter.scatter(measurement[0], measurement[1], **kwargs)
    return ret

def plot_gt(plotter, gt, **kwargs):
    return plot_state(plotter, gt, **kwargs)

def plot_gt_measurement_line(plotter, gt, measurement, **kwargs):
    ret = None
    if measurement != None:
        ret = plotter.plot([gt[0], measurement[0]], [gt[2], measurement[1]], **kwargs)
    return ret

def plot_all_states(plotter, states, **kwargs):
    return plotter.plot([s[0] for s in states], [s[2] for s in states], **kwargs)

def plot_all_state_covs(plotter, state_covariances, states, plot_freq, **kwargs):
    n = len(state_covariances)
    ret = []
    for k in range(n):
        if k % plot_freq == 0:
            ret.append(plot_state_cov(plotter, state_covariances[k], states[k], **kwargs))
    return ret

def plot_all_gts(plotter, gts, **kwargs):
    return plot_all_states(plotter, gts, **kwargs)

def plot_all_measurements(plotter, measurements, **kwargs):
    return plotter.scatter([s[0] for s in measurements if s is not None], [s[1] for s in measurements if s is not None], **kwargs)

def plot_all_gt_measurement_lines(plotter, gts, measurements, plot_freq, **kwargs):
    n = len(gts)
    ret = []
    for k in range(n):
        if k % plot_freq == 0 and measurements[k] is not None:
            ret.append(plotter.plot([gts[k][0], measurements[k][0]], [gts[k][2], measurements[k][1]], **kwargs))
    return ret

"""
 
 #### ##    ## ########  #######  
  ##  ###   ## ##       ##     ## 
  ##  ####  ## ##       ##     ## 
  ##  ## ## ## ######   ##     ## 
  ##  ##  #### ##       ##     ## 
  ##  ##   ### ##       ##     ## 
 #### ##    ## ##        #######  
 
"""

def plot_avg_all_traces(plotter, state_covariances_lists, **kwargs):
    trace_lists = []
    for covs in state_covariances_lists:
        trace_lists.append([np.trace(P) for P in covs])
    mean_traces = np.mean(trace_lists, axis=0)
    return plotter.plot(range(len(mean_traces)), mean_traces, **kwargs)

def plot_avg_all_root_sqr_error(plotter, states_lists, gts_lists, **kwargs):
    diff_lists = []
    for i in range(len(states_lists)):
        states = states_lists[i]
        gts = gts_lists[i]
        diff_lists.append([np.sqrt(x@x) for x in [s-g for s,g in zip(states,gts)]])
    mean_diffs = np.mean(diff_lists, axis=0)
    return plotter.plot(range(len(mean_diffs)), mean_diffs, **kwargs)

def plot_all_traces(plotter, state_covariances, **kwargs):
    traces = [np.trace(P) for P in state_covariances]
    return plotter.plot(range(len(traces)), traces, **kwargs)

def plot_root_sqr_error(plotter, states, gts, **kwargs):
    diff = np.array([np.sqrt(x@x) for x in [s-g for s,g in zip(states,gts)]])
    return plotter.plot(range(len(diff)), diff, **kwargs)

"""
 
 ##     ## ######## ##       ########  ######## ########   ######  
 ##     ## ##       ##       ##     ## ##       ##     ## ##    ## 
 ##     ## ##       ##       ##     ## ##       ##     ## ##       
 ######### ######   ##       ########  ######   ########   ######  
 ##     ## ##       ##       ##        ##       ##   ##         ## 
 ##     ## ##       ##       ##        ##       ##    ##  ##    ## 
 ##     ## ######## ######## ##        ######## ##     ##  ######  
 
"""

# From https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals) # eigvals positive because covariance is positive semi definite
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)