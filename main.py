"""

"""

import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import plotting as plot_funcs
import key_stream as keyed_num_gen
import estimation as est
import privilege_covariances as priv_cov

SIM_TIMESTEPS = 125
NUM_SIMS_TO_AVG = 1000

def main():

    """
 
    888b     d888  .d88888b.  8888888b.  8888888888 888      .d8888b.  
    8888b   d8888 d88P" "Y88b 888  "Y88b 888        888     d88P  Y88b 
    88888b.d88888 888     888 888    888 888        888     Y88b.      
    888Y88888P888 888     888 888    888 8888888    888      "Y888b.   
    888 Y888P 888 888     888 888    888 888        888         "Y88b. 
    888  Y8P  888 888     888 888    888 888        888           "888 
    888   "   888 Y88b. .d88P 888  .d88P 888        888     Y88b  d88P 
    888       888  "Y88888P"  8888888P"  8888888888 88888888 "Y8888P"  
                                                                        
                                                                        
                                                                        
    
    """

    # State dimension
    n = 4

    # Measurement dimension
    m = 2

    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement models
    H1 = np.array([[1, 0, 0, 0], 
                   [0, 0, 1, 0]])
    H2 = np.array([[0, 1, 0, 0], 
                   [0, 0, 0, 1]])

    R = np.array([[5, 2], 
                  [2, 5]])

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]])
    
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 1])

    """
    
    8888888b.  8888888b.  8888888 888     888      8888888888 .d8888b. 88888888888 
    888   Y88b 888   Y88b   888   888     888      888       d88P  Y88b    888     
    888    888 888    888   888   888     888      888       Y88b.         888     
    888   d88P 888   d88P   888   Y88b   d88P      8888888    "Y888b.      888     
    8888888P"  8888888P"    888    Y88b d88P       888           "Y88b.    888     
    888        888 T88b     888     Y88o88P        888             "888    888     
    888        888  T88b    888      Y888P         888       Y88b  d88P    888     
    888        888   T88b 8888888     Y8P          8888888888 "Y8888P"     888     
                                                                                    
                                                                                    
                                                                                    
    
    """

    # Single privilege class parameters
    covar_to_remove = np.array([[35, 0],
                           [0, 35]])

    # Multiple privilege classes parameters
    num_priv_classes = 3
    priv_covars = [np.array([[20, 0],
                             [0, 20]]),
                   np.array([[14, 0],
                             [0, 14]]),
                   np.array([[17, 0],
                             [0, 17]])]
    covars_to_remove = priv_cov.priv_covars_to_covars_to_remove(priv_covars)

    # Encryption for single privilege class
    single_sensor_generator, single_filter_generator = keyed_num_gen.KeyStreamPairFactory.make_shared_key_streams(2)

    # Encryption for multiple privilege classes
    generator_sets = []
    for _ in range(num_priv_classes):
        generator_sets.append(keyed_num_gen.KeyStreamPairFactory.make_shared_key_streams(3))
    sensor_generators, filter_generators, filter_generators_copy = list(zip(*generator_sets))

    """
    
    8888888b.  888      .d88888b. 88888888888      8888888 888b    888 8888888 88888888888 
    888   Y88b 888     d88P" "Y88b    888            888   8888b   888   888       888     
    888    888 888     888     888    888            888   88888b  888   888       888     
    888   d88P 888     888     888    888            888   888Y88b 888   888       888     
    8888888P"  888     888     888    888            888   888 Y88b888   888       888     
    888        888     888     888    888            888   888  Y88888   888       888     
    888        888     Y88b. .d88P    888            888   888   Y8888   888       888     
    888        88888888 "Y88888P"     888          8888888 888    Y888 8888888     888     
                                                                                            
                                                                                            
                                                                                            
    
    """

    # global parameters
    plot_funcs.init_matplotlib_params(True, True)

    """
    
    888888b.    .d88888b.  888     888 888b    888 8888888b.  8888888888 8888888b.  
    888  "88b  d88P" "Y88b 888     888 8888b   888 888  "Y88b 888        888  "Y88b 
    888  .88P  888     888 888     888 88888b  888 888    888 888        888    888 
    8888888K.  888     888 888     888 888Y88b 888 888    888 8888888    888    888 
    888  "Y88b 888     888 888     888 888 Y88b888 888    888 888        888    888 
    888    888 888     888 888     888 888  Y88888 888    888 888        888    888 
    888   d88P Y88b. .d88P Y88b. .d88P 888   Y8888 888  .d88P 888        888  .d88P 
    8888888P"   "Y88888P"   "Y88888P"  888    Y888 8888888P"  8888888888 8888888P"  
                                                                                    
                                                                                    
                                                                                    
    
    """

    all_sim_gts = []
    all_sim_ys = []

    all_sim_unpriv_pred_lists = []
    all_sim_priv_pred_lists = []
    
    all_sim_unpriv_upd_lists = []
    all_sim_priv_upd_lists = []

    for i in range(NUM_SIMS_TO_AVG):
        print("Running bounded sim %d..." % (i+1))

        """

         ######## #### ##       ######## ######## ########     #### ##    ## #### ######## 
         ##        ##  ##          ##    ##       ##     ##     ##  ###   ##  ##     ##    
         ##        ##  ##          ##    ##       ##     ##     ##  ####  ##  ##     ##    
         ######    ##  ##          ##    ######   ########      ##  ## ## ##  ##     ##    
         ##        ##  ##          ##    ##       ##   ##       ##  ##  ####  ##     ##    
         ##        ##  ##          ##    ##       ##    ##      ##  ##   ###  ##     ##    
         ##       #### ########    ##    ######## ##     ##    #### ##    ## ####    ##    

        """

        # Bounded filters
        unpriv_filter_bounded = est.UnprivFilter(n, m, F, Q, H1, R, init_state, init_cov, [covar_to_remove])
        priv_filter_bounded = est.PrivFilter(n, m, F, Q, H1, R, init_state, init_cov, np.zeros((2,2)), covar_to_remove, single_filter_generator)

        # Sensor
        sensor_bounded = est.SensorWithPrivileges(n, m, H1, R, [covar_to_remove], [single_sensor_generator])

        # Ground truth
        ground_truth = est.GroundTruth(F, Q, gt_init_state)

        """

         ########  ##     ## ##    ##    ########     ###    ########    ###    
         ##     ## ##     ## ###   ##    ##     ##   ## ##      ##      ## ##   
         ##     ## ##     ## ####  ##    ##     ##  ##   ##     ##     ##   ##  
         ########  ##     ## ## ## ##    ##     ## ##     ##    ##    ##     ## 
         ##   ##   ##     ## ##  ####    ##     ## #########    ##    ######### 
         ##    ##  ##     ## ##   ###    ##     ## ##     ##    ##    ##     ## 
         ##     ##  #######  ##    ##    ########  ##     ##    ##    ##     ## 

        """

        gts = []
        ys = []

        unpriv_pred_list = []
        priv_pred_list = []

        unpriv_upd_list = []
        priv_upd_list = []
        """

          ######  #### ##     ##    ##        #######   #######  ########  
         ##    ##  ##  ###   ###    ##       ##     ## ##     ## ##     ## 
         ##        ##  #### ####    ##       ##     ## ##     ## ##     ## 
          ######   ##  ## ### ##    ##       ##     ## ##     ## ########  
               ##  ##  ##     ##    ##       ##     ## ##     ## ##        
         ##    ##  ##  ##     ##    ##       ##     ## ##     ## ##        
          ######  #### ##     ##    ########  #######   #######  ##        

        """

        for _ in range(SIM_TIMESTEPS):
            gt = ground_truth.update()
            y = sensor_bounded.measure(gt)

            # Predict
            unpriv_pred = unpriv_filter_bounded.predict()
            priv_pred = priv_filter_bounded.predict()
            
            # Update
            upriv_upd = unpriv_filter_bounded.update(y)
            priv_upd = priv_filter_bounded.update(y)
            
            # Save run data
            gts.append(gt)
            ys.append(y)

            unpriv_pred_list.append(unpriv_pred)
            priv_pred_list.append(priv_pred)
            
            unpriv_upd_list.append(upriv_upd)
            priv_upd_list.append(priv_upd)
        

        """

          ######     ###    ##     ## ########    ########     ###    ########    ###    
         ##    ##   ## ##   ##     ## ##          ##     ##   ## ##      ##      ## ##   
         ##        ##   ##  ##     ## ##          ##     ##  ##   ##     ##     ##   ##  
          ######  ##     ## ##     ## ######      ##     ## ##     ##    ##    ##     ## 
               ## #########  ##   ##  ##          ##     ## #########    ##    ######### 
         ##    ## ##     ##   ## ##   ##          ##     ## ##     ##    ##    ##     ## 
          ######  ##     ##    ###    ########    ########  ##     ##    ##    ##     ## 

        """

        all_sim_gts.append(gts)
        all_sim_ys.append(ys)

        all_sim_unpriv_pred_lists.append(unpriv_pred_list)
        all_sim_priv_pred_lists.append(priv_pred_list)

        all_sim_unpriv_upd_lists.append(unpriv_upd_list)
        all_sim_priv_upd_lists.append(priv_upd_list)

    """
    
    ########  ##        #######  ######## 
    ##     ## ##       ##     ##    ##    
    ##     ## ##       ##     ##    ##    
    ########  ##       ##     ##    ##    
    ##        ##       ##     ##    ##    
    ##        ##       ##     ##    ##    
    ##        ########  #######     ##    
    
    """

    # Bounded plot
    print("Making single level bounded plot ...")

    plot_width = 3.4
    plot_height = 1.7
    fig_bounded = plt.figure()
    fig_bounded.set_size_inches(w=plot_width, h=plot_height)

    ax_tr_bounded = fig_bounded.add_subplot(121)
    ax_tr_bounded.set_title(r"Error Covariance Traces")
    ax_tr_bounded.set_xlabel(r"Simulation Time")
    ax_tr_bounded.set_ylabel(r"Trace")
    ax_tr_bounded.set_xticks([])

    ax_rmse_bounded = fig_bounded.add_subplot(122)
    ax_rmse_bounded.set_title(r"Errors")
    ax_rmse_bounded.set_xlabel(r"Simulation Time")
    ax_rmse_bounded.set_ylabel(r"RMSE")
    ax_rmse_bounded.set_xticks([])

    # Size adjusting to fit in latex columns
    plt.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.7, wspace=0.4)

    diff_legend, = plot_funcs.plot_avg_all_trace_diffs(ax_tr_bounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], 
                                                                      [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], 
                                                                      linestyle='--', color='darkgray')
    unpriv_legend, = plot_funcs.plot_avg_all_traces(ax_tr_bounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], linestyle='-', color='darkred')
    priv_legend, = plot_funcs.plot_avg_all_traces(ax_tr_bounded, [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], linestyle='-', color='darkgreen')
    

    plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_bounded, [[s[0] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], all_sim_gts, linestyle='-', color='darkred')
    plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_bounded, [[s[0] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], all_sim_gts, linestyle='-', color='darkgreen')

    fig_bounded.legend(handles=[priv_legend, unpriv_legend, diff_legend], 
               labels=["Priv.", "Unpriv.", "Diff."],
               loc="upper center",
               ncol=3)
    
    # Save or show figure
    if matplotlib.get_backend() == 'pgf':
        plt.savefig('pictures/single_level_bounded.pdf')
    else:
        plt.show()
    
    """
    
    888     888 888b    888 888888b.    .d88888b.  888     888 888b    888 8888888b.  8888888888 8888888b.  
    888     888 8888b   888 888  "88b  d88P" "Y88b 888     888 8888b   888 888  "Y88b 888        888  "Y88b 
    888     888 88888b  888 888  .88P  888     888 888     888 88888b  888 888    888 888        888    888 
    888     888 888Y88b 888 8888888K.  888     888 888     888 888Y88b 888 888    888 8888888    888    888 
    888     888 888 Y88b888 888  "Y88b 888     888 888     888 888 Y88b888 888    888 888        888    888 
    888     888 888  Y88888 888    888 888     888 888     888 888  Y88888 888    888 888        888    888 
    Y88b. .d88P 888   Y8888 888   d88P Y88b. .d88P Y88b. .d88P 888   Y8888 888  .d88P 888        888  .d88P 
    "Y88888P"  888    Y888 8888888P"   "Y88888P"   "Y88888P"  888    Y888 8888888P"  8888888888 8888888P"  
                                                                                                            
                                                                                                            
                                                                                                            
    
    """

    all_sim_gts = []
    all_sim_ys = []

    all_sim_unpriv_pred_lists = []
    all_sim_priv_pred_lists = []
    
    all_sim_unpriv_upd_lists = []
    all_sim_priv_upd_lists = []

    for i in range(NUM_SIMS_TO_AVG):
        print("Running unbounded sim %d..." % (i+1))

        """

         ######## #### ##       ######## ######## ########     #### ##    ## #### ######## 
         ##        ##  ##          ##    ##       ##     ##     ##  ###   ##  ##     ##    
         ##        ##  ##          ##    ##       ##     ##     ##  ####  ##  ##     ##    
         ######    ##  ##          ##    ######   ########      ##  ## ## ##  ##     ##    
         ##        ##  ##          ##    ##       ##   ##       ##  ##  ####  ##     ##    
         ##        ##  ##          ##    ##       ##    ##      ##  ##   ###  ##     ##    
         ##       #### ########    ##    ######## ##     ##    #### ##    ## ####    ##    

        """

        # Unbounded filters
        unpriv_filter_unbounded = est.UnprivFilter(n, m, F, Q, H2, R, init_state, init_cov, [covar_to_remove])
        priv_filter_unbounded = est.PrivFilter(n, m, F, Q, H2, R, init_state, init_cov, np.zeros((2,2)), covar_to_remove, single_filter_generator)

        # Sensor
        sensor_unbounded = est.SensorWithPrivileges(n, m, H2, R, [covar_to_remove], [single_sensor_generator])

        # Ground truth
        ground_truth = est.GroundTruth(F, Q, gt_init_state)

        """

         ########  ##     ## ##    ##    ########     ###    ########    ###    
         ##     ## ##     ## ###   ##    ##     ##   ## ##      ##      ## ##   
         ##     ## ##     ## ####  ##    ##     ##  ##   ##     ##     ##   ##  
         ########  ##     ## ## ## ##    ##     ## ##     ##    ##    ##     ## 
         ##   ##   ##     ## ##  ####    ##     ## #########    ##    ######### 
         ##    ##  ##     ## ##   ###    ##     ## ##     ##    ##    ##     ## 
         ##     ##  #######  ##    ##    ########  ##     ##    ##    ##     ## 

        """

        gts = []
        ys = []

        unpriv_pred_list = []
        priv_pred_list = []

        unpriv_upd_list = []
        priv_upd_list = []

        """

          ######  #### ##     ##    ##        #######   #######  ########  
         ##    ##  ##  ###   ###    ##       ##     ## ##     ## ##     ## 
         ##        ##  #### ####    ##       ##     ## ##     ## ##     ## 
          ######   ##  ## ### ##    ##       ##     ## ##     ## ########  
               ##  ##  ##     ##    ##       ##     ## ##     ## ##        
         ##    ##  ##  ##     ##    ##       ##     ## ##     ## ##        
          ######  #### ##     ##    ########  #######   #######  ##        

        """

        for _ in range(SIM_TIMESTEPS):
            gt = ground_truth.update()
            y = sensor_unbounded.measure(gt)

            # Predict
            unpriv_pred = unpriv_filter_unbounded.predict()
            priv_pred = priv_filter_unbounded.predict()
            
            # Update
            upriv_upd = unpriv_filter_unbounded.update(y)
            priv_upd = priv_filter_unbounded.update(y)
            
            # Save run data
            gts.append(gt)
            ys.append(y)

            unpriv_pred_list.append(unpriv_pred)
            priv_pred_list.append(priv_pred)
            
            unpriv_upd_list.append(upriv_upd)
            priv_upd_list.append(priv_upd)
        

        """

          ######     ###    ##     ## ########    ########     ###    ########    ###    
         ##    ##   ## ##   ##     ## ##          ##     ##   ## ##      ##      ## ##   
         ##        ##   ##  ##     ## ##          ##     ##  ##   ##     ##     ##   ##  
          ######  ##     ## ##     ## ######      ##     ## ##     ##    ##    ##     ## 
               ## #########  ##   ##  ##          ##     ## #########    ##    ######### 
         ##    ## ##     ##   ## ##   ##          ##     ## ##     ##    ##    ##     ## 
          ######  ##     ##    ###    ########    ########  ##     ##    ##    ##     ## 

        """

        all_sim_gts.append(gts)
        all_sim_ys.append(ys)

        all_sim_unpriv_pred_lists.append(unpriv_pred_list)
        all_sim_priv_pred_lists.append(priv_pred_list)

        all_sim_unpriv_upd_lists.append(unpriv_upd_list)
        all_sim_priv_upd_lists.append(priv_upd_list)

    """
    
    ########  ##        #######  ######## 
    ##     ## ##       ##     ##    ##    
    ##     ## ##       ##     ##    ##    
    ########  ##       ##     ##    ##    
    ##        ##       ##     ##    ##    
    ##        ##       ##     ##    ##    
    ##        ########  #######     ##    
    
    """

    # Unbounded plot
    print("Making single level unbounded plot ...")

    plot_width = 3.4
    plot_height = 1.7
    fig_unbounded = plt.figure()
    fig_unbounded.set_size_inches(w=plot_width, h=plot_height)

    ax_tr_unbounded = fig_unbounded.add_subplot(121)
    ax_tr_unbounded.set_title(r"Error Covariance Traces")
    ax_tr_unbounded.set_xlabel(r"Simulation Time")
    ax_tr_unbounded.set_ylabel(r"Trace")
    ax_tr_unbounded.set_xticks([])

    ax_rmse_unbounded = fig_unbounded.add_subplot(122)
    ax_rmse_unbounded.set_title(r"Errors")
    ax_rmse_unbounded.set_xlabel(r"Simulation Time")
    ax_rmse_unbounded.set_ylabel(r"RMSE")
    ax_rmse_unbounded.set_xticks([])

    # Size adjusting to fit in latex columns
    plt.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.7, wspace=0.4)

    diff_legend, = plot_funcs.plot_avg_all_trace_diffs(ax_tr_unbounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], 
                                                                        [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], 
                                                                        linestyle='--', color='darkgray')
    unpriv_legend, = plot_funcs.plot_avg_all_traces(ax_tr_unbounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], linestyle='-', color='darkred')
    priv_legend, = plot_funcs.plot_avg_all_traces(ax_tr_unbounded, [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], linestyle='-', color='darkgreen')
    

    plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_unbounded, [[s[0] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], all_sim_gts, linestyle='-', color='darkred')
    plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_unbounded, [[s[0] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], all_sim_gts, linestyle='-', color='darkgreen')

    fig_unbounded.legend(handles=[priv_legend, unpriv_legend, diff_legend], 
               labels=["Priv.", "Unpriv.", "Diff."],
               loc="upper center",
               ncol=3)
    
    # Save or show figure
    if matplotlib.get_backend() == 'pgf':
        plt.savefig('pictures/single_level_unbounded.pdf')
    else:
        plt.show()

    """
    
    888b     d888 888     888 888    88888888888 8888888 8888888b.  888      8888888888 
    8888b   d8888 888     888 888        888       888   888   Y88b 888      888        
    88888b.d88888 888     888 888        888       888   888    888 888      888        
    888Y88888P888 888     888 888        888       888   888   d88P 888      8888888    
    888 Y888P 888 888     888 888        888       888   8888888P"  888      888        
    888  Y8P  888 888     888 888        888       888   888        888      888        
    888   "   888 Y88b. .d88P 888        888       888   888        888      888        
    888       888  "Y88888P"  88888888   888     8888888 888        88888888 8888888888 
                                                                                        
                                                                                        
                                                                                        
    
    """

    all_sim_gts = []
    all_sim_ys = []

    all_sim_unpriv_pred_lists = []
    all_sim_all_key_priv_pred_lists = []
    all_sim_priv_preds_lists = []
    
    all_sim_unpriv_upd_lists = []
    all_sim_all_key_priv_upd_lists = []
    all_sim_priv_upds_lists = []

    for i in range(NUM_SIMS_TO_AVG):
        print("Running multiple sim %d..." % (i+1))

        """

         ######## #### ##       ######## ######## ########     #### ##    ## #### ######## 
         ##        ##  ##          ##    ##       ##     ##     ##  ###   ##  ##     ##    
         ##        ##  ##          ##    ##       ##     ##     ##  ####  ##  ##     ##    
         ######    ##  ##          ##    ######   ########      ##  ## ## ##  ##     ##    
         ##        ##  ##          ##    ##       ##   ##       ##  ##  ####  ##     ##    
         ##        ##  ##          ##    ##       ##    ##      ##  ##   ###  ##     ##    
         ##       #### ########    ##    ######## ##     ##    #### ##    ## ####    ##    

        """
        
        # Multiple privilege filters
        unpriv_filter = est.UnprivFilter(n, m, F, Q, H1, R, init_state, init_cov, covars_to_remove)
        all_key_priv_filter = est.MultKeyPrivFilter(n, m, F, Q, H1, R, init_state, init_cov, np.zeros((2,2)), covars_to_remove, filter_generators_copy)
        priv_filters = []
        for i in range(num_priv_classes):
            priv_filters.append(est.PrivFilter(n, m, F, Q, H1, R, init_state, init_cov, priv_covars[i], covars_to_remove[i], filter_generators[i]))

        # Sensor
        sensor_mult = est.SensorWithPrivileges(n, m, H1, R, covars_to_remove, sensor_generators)

        # Ground truth
        ground_truth = est.GroundTruth(F, Q, gt_init_state)

        """

         ########  ##     ## ##    ##    ########     ###    ########    ###    
         ##     ## ##     ## ###   ##    ##     ##   ## ##      ##      ## ##   
         ##     ## ##     ## ####  ##    ##     ##  ##   ##     ##     ##   ##  
         ########  ##     ## ## ## ##    ##     ## ##     ##    ##    ##     ## 
         ##   ##   ##     ## ##  ####    ##     ## #########    ##    ######### 
         ##    ##  ##     ## ##   ###    ##     ## ##     ##    ##    ##     ## 
         ##     ##  #######  ##    ##    ########  ##     ##    ##    ##     ## 

        """

        gts = []
        ys = []

        unpriv_pred_list = []
        all_key_priv_pred_list = []
        priv_preds_lists = []

        unpriv_upd_list = []
        all_key_priv_upd_list = []
        priv_upds_lists = []

        """

          ######  #### ##     ##    ##        #######   #######  ########  
         ##    ##  ##  ###   ###    ##       ##     ## ##     ## ##     ## 
         ##        ##  #### ####    ##       ##     ## ##     ## ##     ## 
          ######   ##  ## ### ##    ##       ##     ## ##     ## ########  
               ##  ##  ##     ##    ##       ##     ## ##     ## ##        
         ##    ##  ##  ##     ##    ##       ##     ## ##     ## ##        
          ######  #### ##     ##    ########  #######   #######  ##        

        """

        for _ in range(SIM_TIMESTEPS):
            gt = ground_truth.update()
            y = sensor_mult.measure(gt)

            # Predict
            unpriv_pred = unpriv_filter.predict()
            all_key_priv_pred = all_key_priv_filter.predict()
            priv_preds = []
            for i in range(num_priv_classes):
                priv_preds.append(priv_filters[i].predict())
            
            # Update
            upriv_upd = unpriv_filter.update(y)
            all_key_priv_upd = all_key_priv_filter.update(y)
            priv_upds = []
            for i in range(num_priv_classes):
                priv_upds.append(priv_filters[i].update(y))
            
            # Save run data
            gts.append(gt)
            ys.append(y)

            unpriv_pred_list.append(unpriv_pred)
            all_key_priv_pred_list.append(all_key_priv_pred)
            priv_preds_lists.append(copy.deepcopy(priv_preds))
            
            unpriv_upd_list.append(upriv_upd)
            all_key_priv_upd_list.append(all_key_priv_upd)
            priv_upds_lists.append(copy.deepcopy(priv_upds))
        

        """

          ######     ###    ##     ## ########    ########     ###    ########    ###    
         ##    ##   ## ##   ##     ## ##          ##     ##   ## ##      ##      ## ##   
         ##        ##   ##  ##     ## ##          ##     ##  ##   ##     ##     ##   ##  
          ######  ##     ## ##     ## ######      ##     ## ##     ##    ##    ##     ## 
               ## #########  ##   ##  ##          ##     ## #########    ##    ######### 
         ##    ## ##     ##   ## ##   ##          ##     ## ##     ##    ##    ##     ## 
          ######  ##     ##    ###    ########    ########  ##     ##    ##    ##     ## 

        """

        all_sim_gts.append(gts)
        all_sim_ys.append(ys)

        all_sim_unpriv_pred_lists.append(unpriv_pred_list)
        all_sim_all_key_priv_pred_lists.append(all_key_priv_pred_list)
        all_sim_priv_preds_lists.append(copy.deepcopy(priv_preds_lists))

        all_sim_unpriv_upd_lists.append(unpriv_upd_list)
        all_sim_all_key_priv_upd_lists.append(all_key_priv_upd_list)
        all_sim_priv_upds_lists.append(copy.deepcopy(priv_upds_lists))

    
    # # Start sim
    # for _ in range(SIM_TIMESTEPS):
    #     gt = ground_truth.update()
    #     y = sensor.measure(gt)

    #     # Predict
    #     unpriv_pred = unpriv_filter.predict()
    #     all_key_priv_pred = all_key_priv_filter.predict()
    #     priv_preds = []
    #     for i in range(num_priv_classes):
    #         priv_preds.append(priv_filters[i].predict())

    #     # Update
    #     upriv_upd = unpriv_filter.update(y)
    #     all_key_priv_upd = all_key_priv_filter.update(y)
    #     priv_upds = []
    #     for i in range(num_priv_classes):
    #         priv_upds.append(priv_filters[i].update(y))

    #     # Save all data
    #     gts.append(gt)
    #     ys.append(y)
    #     unpriv_pred_list.append(unpriv_pred)
    #     all_key_priv_pred_list.append(all_key_priv_pred)
    #     for i in range(num_priv_classes):
    #         priv_pred_lists[i].append(priv_preds[i])
    #     unpriv_upd_list.append(upriv_upd)
    #     all_key_priv_upd_list.append(all_key_priv_upd)
    #     for i in range(num_priv_classes):
    #         priv_upd_lists[i].append(priv_upds[i])

    """
    
    ########  ##        #######  ######## 
    ##     ## ##       ##     ##    ##    
    ##     ## ##       ##     ##    ##    
    ########  ##       ##     ##    ##    
    ##        ##       ##     ##    ##    
    ##        ##       ##     ##    ##    
    ##        ########  #######     ##    
    
    """

    # Multiple level plot
    print("Making multiple level plot ...")

    plot_width = 3.4
    plot_height = 1.7
    fig_mult = plt.figure()
    fig_mult.set_size_inches(w=plot_width, h=plot_height)

    ax_tr_mult = fig_mult.add_subplot(121)
    ax_tr_mult.set_title(r"Error Covariance Traces")
    ax_tr_mult.set_xlabel(r"Simulation Time")
    ax_tr_mult.set_ylabel(r"Trace")
    ax_tr_mult.set_xticks([])

    ax_rmse_mult = fig_mult.add_subplot(122)
    ax_rmse_mult.set_title(r"Errors")
    ax_rmse_mult.set_xlabel(r"Simulation Time")
    ax_rmse_mult.set_ylabel(r"RMSE")
    ax_rmse_mult.set_xticks([])

    # Size adjusting to fit in latex columns
    plt.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.6, wspace=0.4)

    unpriv_legend, = plot_funcs.plot_avg_all_traces(ax_tr_mult, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], linestyle='--', color='darkred')
    all_key_priv_legend, = plot_funcs.plot_avg_all_traces(ax_tr_mult, [[s[1] for s in ak_pr_upd_l] for ak_pr_upd_l in all_sim_all_key_priv_upd_lists], linestyle='--', color='darkgreen')

    priv_cs = []
    priv_legends = []
    for i in range(num_priv_classes):
        c = 'C'+str(i)
        all_sim_priv_upd_list = [[x[i] for x in y]for y in all_sim_priv_upds_lists]
        priv_cs.append(c)

        legend, = plot_funcs.plot_avg_all_traces(ax_tr_mult, [[s[1] for s in upd_l] for upd_l in all_sim_priv_upd_list], color=c)
        plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_mult, [[s[0] for s in upd_l] for upd_l in all_sim_priv_upd_list], all_sim_gts, color=c)

        priv_legends.append(legend)
    

    plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_mult, [[s[0] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], all_sim_gts, linestyle='--', color='darkred')
    plot_funcs.plot_avg_all_root_sqr_error(ax_rmse_mult, [[s[0] for s in pr_upd_l] for pr_upd_l in all_sim_all_key_priv_upd_lists], all_sim_gts, linestyle='--', color='darkgreen')

    fig_mult.legend(handles=[all_key_priv_legend, unpriv_legend]+priv_legends, 
               labels=["All Key", "No Key"] + ["Priv. %d" %(i+1) for i in range(num_priv_classes)],
               loc="upper center",
               ncol=3)
    
    # Save or show figure
    if matplotlib.get_backend() == 'pgf':
        plt.savefig('pictures/multiple_level.pdf')
    else:
        plt.show()

    # # Unprivileged estimation plots
    # unpriv_c = 'darkred'
    # unpriv_legend, = plot_funcs.plot_all_states(ax, [s[0] for s in unpriv_upd_list], linestyle='--', color=unpriv_c)
    # plot_funcs.plot_all_state_covs(ax, [s[1] for s in unpriv_upd_list], [s[0] for s in unpriv_upd_list], 10, fill=False, linestyle='--', edgecolor=unpriv_c)
    # plot_funcs.plot_all_traces(ax2, [s[1] for s in unpriv_upd_list], linestyle='--', color=unpriv_c)
    # plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in unpriv_upd_list], gts, linestyle='--', color=unpriv_c)

    # # All key privileged estimation plots
    # all_key_priv_c = 'darkgreen'
    # all_key_priv_legend, = plot_funcs.plot_all_states(ax, [s[0] for s in all_key_priv_upd_list], linestyle='--', color=all_key_priv_c)
    # plot_funcs.plot_all_state_covs(ax, [s[1] for s in all_key_priv_upd_list], [s[0] for s in all_key_priv_upd_list], 10, fill=False, linestyle='--', edgecolor=all_key_priv_c)
    # plot_funcs.plot_all_traces(ax2, [s[1] for s in all_key_priv_upd_list], linestyle='--', color=all_key_priv_c)
    # plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in all_key_priv_upd_list], gts, linestyle='--', color=all_key_priv_c)

    # # Privileged estimation plots
    # priv_cs = []
    # priv_legends = []
    # for i in range(num_priv_classes):
    #     c = 'C'+str(i)
    #     priv_update_list = priv_upd_lists[i]
    #     priv_cs.append(c)

    #     priv_legends.append(plot_funcs.plot_all_states(ax, [s[0] for s in priv_update_list], color=c)[0])
    #     plot_funcs.plot_all_state_covs(ax, [s[1] for s in priv_update_list], [s[0] for s in priv_update_list], 10, fill=False, linestyle='-', edgecolor=c)
    #     plot_funcs.plot_all_traces(ax2, [s[1] for s in priv_update_list], color=c)
    #     plot_funcs.plot_root_sqr_error(ax3, [s[0] for s in priv_update_list], gts, color=c)
    
    # # Shared legend
    # fig.legend(handles=[gt_legend, m_legend, unpriv_legend, all_key_priv_legend]+priv_legends, 
    #            labels=["Ground Truth", "Measurements", "No Key Estimator", "All Key Estimator"]+["Privileged Estimator "+str(i+1) for i in range(num_priv_classes)],
    #            loc="upper center",
    #            ncol=2)



if __name__ == '__main__':
    main()