"""

"""

import numpy as np

"""
 
  ######   ######## 
 ##    ##     ##    
 ##           ##    
 ##   ####    ##    
 ##    ##     ##    
 ##    ##     ##    
  ######      ##    
 
"""

class GroundTruth:
    def __init__(self, F, Q, init_state):
        self.F = F
        self.Q = Q
        self.state = init_state
        return
    
    def update(self):
        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), self.Q)
        self.state = self.F@self.state + w
        return self.state

"""
 
  ######  ######## ##    ##  ######   #######  ########  
 ##    ## ##       ###   ## ##    ## ##     ## ##     ## 
 ##       ##       ####  ## ##       ##     ## ##     ## 
  ######  ######   ## ## ##  ######  ##     ## ########  
       ## ##       ##  ####       ## ##     ## ##   ##   
 ##    ## ##       ##   ### ##    ## ##     ## ##    ##  
  ######  ######## ##    ##  ######   #######  ##     ## 
 
"""

class SensorAbs:
    def measure(self, ground_truth):
        raise NotImplementedError

class SensorPure(SensorAbs):
    def __init__(self, n, H, R):
        self.n = n
        self.H = H
        self.R = R
        return
    
    def measure(self, ground_truth):
        v = np.random.multivariate_normal(np.array([0, 0]), self.R)
        return self.H@ground_truth + v

class SensorWithPrivileges(SensorPure):
    def __init__(self, n, H, R, add_covars, generators):
        assert (len(add_covars) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
        super().__init__(n, H, R)
        self.add_covars = add_covars
        self.generators = generators
        self.num_privs = len(add_covars)
        return
    
    def measure(self, ground_truth):
        return super().measure(ground_truth) + self.get_sum_of_additional_noises()
    
    def get_sum_of_additional_noises(self):
        noise = 0
        for i in range(self.num_privs):
            noise += self.generators[i].next_n_as_gaussian(self.n, np.array([0 for _ in range(self.n)]), self.add_covars[i])
        return noise
    

"""
 
 ######## #### ##       ######## ######## ########   ######  
 ##        ##  ##          ##    ##       ##     ## ##    ## 
 ##        ##  ##          ##    ##       ##     ## ##       
 ######    ##  ##          ##    ######   ########   ######  
 ##        ##  ##          ##    ##       ##   ##         ## 
 ##        ##  ##          ##    ##       ##    ##  ##    ## 
 ##       #### ########    ##    ######## ##     ##  ######  
 
"""

class FilterAbs:
    def predict(self):
        raise NotImplementedError
    
    def update(self, measurement):
        raise NotImplementedError

class KFilter(FilterAbs):
    def __init__(self, n, F, Q, H, R, init_state, init_cov):
        self.n = n
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        return
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return self.x, self.P
    
    def update(self, measurement):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return self.x, self.P

class UnprivFilter(KFilter):
    pass

class PrivFilter(KFilter):
    def __init__(self, n, F, Q, H, R, init_state, init_cov, add_covar, generator):
        super().__init__(n, F, Q, H, R, init_state, init_cov)
        self.add_covar = add_covar
        self.generator = generator
        return
    
    def update(self, measurement):
        super().update(measurement - self.get_additional_noise())
        return self.x, self.P
    
    def get_additional_noise(self):
        return self.generator.next_n_as_gaussian(self.n, np.array([0 for _ in range(self.n)]), self.add_covar)

class MultKeyPrivFilter(KFilter):
    def __init__(self, n, F, Q, H, R, init_state, init_cov, add_covars, generators):
        assert (len(add_covars) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
        super().__init__(n, F, Q, H, R, init_state, init_cov)
        self.add_covars = add_covars
        self.generators = generators
        self.num_privs = len(add_covars)
        return
    
    def update(self, measurement):
        super().update(measurement - self.get_sum_of_additional_noises())
        return self.x, self.P
    
    def get_sum_of_additional_noises(self):
        noise = 0
        for i in range(self.num_privs):
            noise += self.generators[i].next_n_as_gaussian(self.n, np.array([0 for _ in range(self.n)]), self.add_covars[i])
        return noise