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

def init_matplotlib_params(show_latex_fig):
    fontsize = 9
    linewidth = 1.0
    gridlinewidth = 0.7

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

    if show_latex_fig:
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
    plotter.scatter(state[0], state[2], **kwargs)
    return

def plot_state_cov(plotter, state_covariance, state, **kwargs):
    state_2d = np.array([state[0], state[2]])
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[2][0]],
                             [state_covariance[0][2], state_covariance[2][2]]])
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    plotter.add_artist(ellipse)
    return

def plot_measurement(plotter, measurement, **kwargs):
    if measurement != None:
        plotter.scatter(measurement[0], measurement[1], **kwargs)
    return

def plot_gt(plotter, gt, **kwargs):
    plot_state(plotter, gt, **kwargs)
    return

def plot_gt_measurement_line(plotter, gt, measurement, **kwargs):
    if measurement != None:
        plotter.plot([gt[0], measurement[0]], [gt[2], measurement[1]], **kwargs)
    return

def plot_all_states(plotter, states, **kwargs):
    plotter.plot([s[0] for s in states], [s[2] for s in states], **kwargs)
    return

def plot_all_state_covs(plotter, state_covariances, states, plot_freq, **kwargs):
    n = len(state_covariances)
    for k in range(n):
        if k % plot_freq == 0:
            plot_state_cov(plotter, state_covariances[k], states[k], **kwargs)
    return

def plot_all_gts(plotter, gts, **kwargs):
    plot_all_states(plotter, gts, **kwargs)
    return

def plot_all_measurements(plotter, measurements, **kwargs):
    plotter.scatter([s[0] for s in measurements if s is not None], [s[1] for s in measurements if s is not None], **kwargs)
    return

def plot_all_gt_measurement_lines(plotter, gts, measurements, plot_freq, **kwargs):
    n = len(gts)
    for k in range(n):
        if k % plot_freq == 0 and measurements[k] is not None:
            plotter.plot([gts[k][0], measurements[k][0]], [gts[k][2], measurements[k][1]], **kwargs)
    return

"""
 
 #### ##    ## ########  #######  
  ##  ###   ## ##       ##     ## 
  ##  ####  ## ##       ##     ## 
  ##  ## ## ## ######   ##     ## 
  ##  ##  #### ##       ##     ## 
  ##  ##   ### ##       ##     ## 
 #### ##    ## ##        #######  
 
"""

def plot_event_curve(plotter, f, min_measurement, max_measurement, **kwargs):
    diff = max_measurement - min_measurement
    ys = np.linspace(min_measurement-diff, max_measurement+diff, 1000)
    plotter.plot(ys, [f(y) for y in ys], **kwargs)
    plotter.plot(2*[min_measurement], [0,1], color='gray')
    plotter.plot(2*[max_measurement], [0,1], color='gray')
    return

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