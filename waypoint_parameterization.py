import time
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import sys

class ParametricWaypoint():
    # https://en.wikibooks.org/wiki/Calculus/Integration_techniques/Trigonometric_Substitution
    # https://math.stackexchange.com/questions/3851768/moving-a-point-along-a-polynomial-curve-by-certain-arc-length
    # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter19.04-Newton-Raphson-Method.html

    def __init__(self, logger=None):
        self.test_needed = True
        self.equidistance_method = 0
        self.k = 0
        self.param = np.array([0., 0., 0.])
        self.tdalogger = logger
        
        ######## TESTING PARAMS #######
        self.max_dist_wp_x = 4.0 #meter
        self.max_std_dev_curve = 0.6 
        self.verbose = False
    
    def lane_model(self, x):
        A = np.array([x**2, x, np.ones(len(x))], dtype="float").T
        return A 
    
    def calculate_variance(self, data):
        Y = data[:,1]
        param, A = self.obtain_param(data)
        Y_pred = A@param
        return np.std(Y-Y_pred), Y, Y_pred
    
    def obtain_param(self, data):
        ## obtain the coefficient of the curves
        x = data[:,0]
        Y = data[:,1]
        A = self.lane_model(x)
        self.param = np.linalg.pinv(A.T@A)@A.T@Y
        
        self.len_points = self.integral_x(x[-1]) - self.integral_x(x[0])
        # print("len : ", self.len_points)
        return self.param, A
        
    def bisect(self, ds, x0):
        a = self.param[0]
        b = self.param[1]
        c = self.param[2]
        dy = 2*a*x0+b
        self.k = 4*a*ds + np.arcsinh(dy) + (dy)*np.sqrt(1+(dy)**2)
        x0 = np.sqrt(np.sign(self.k)*self.k)
        try:
            root = optimize.newton(self.solve_x, x0, disp = True, fprime = lambda x:2*(x**2 + 1)**(1/2), fprime2 = lambda x:(2*x)/(x**2 + 1)**(1/2))
        except:
            raise ValueError("Optimization Failed") 
        x = (root - b)/(2*a)
        y = a*x**2 + b*x + c
        theta = np.arctan(2*a*x+b) - np.arctan(b)
        return [x,y,theta]  
        
    def integral_x(self,x):
        a = self.param[0]
        b = self.param[1]
        return (np.arcsinh(2*a*x+ b) + (2*a*x+ b)*(1 + (2*a*x+ b)**2)**(1/2))/(4*a) #only one real root at x = 1

    def solve_x(self,x):
        return (np.arcsinh(x) + x*np.sqrt(1+x**2) - self.k)
    
    def test_waypoints(self, wp, all_test = True):
        wp = np.asarray(wp)
        if all_test == True:
            ## test for number of points
            if len(wp) < 2:
                print ("Not enough points")
                return False

            ## test for x order
            for i in range(len(wp)-1):
                if wp[i+1,0] < wp[i,0]:
                    # print ("fails in order test")
                    return False
                
            ## test for repeatation
            for i in range(len(wp)-1):
                if wp[i+1,0] == wp[i,0]:
                    print ("fails in repeatation test")
                    return False
                        
            ## test how far is the first point (reject if its too far)
            if wp[0,0] > self.max_dist_wp_x:
                print ("wp[0,0], self.max_dist_wp_x", wp[0,0], self.max_dist_wp_x)
                print ("fails in first point distance test")
                return False
            
            ## test for variance after fitting to curve
            self.lane_model(wp[:,0])
            var, Y, Y_pred = self.calculate_variance(wp)
            if self.verbose:
                print ("Fitting std_dev = ", var)
            if var > self.max_std_dev_curve:
                print ("fails in standard deviation test, std_dev = ", var)
                return False

        else:
            ## test for variance after fitting to curve
            self.lane_model(wp[:,0])
            var, Y, Y_pred = self.calculate_variance(wp)
            if self.verbose:
                print ("Fitting std_dev = ", var)
            if var > self.max_std_dev_curve:
                print ("fails in standard deviation test, std_dev = ", var)
                return False

        return True
        
    def get_N_points(self, start_x, ds, N = 10):
        pts_hist = []
        start_y = (self.lane_model(np.array([[start_x]]))@self.param)[0]
        # pts_hist.append([start_x, start_y])
        x = start_x
        
        for i in range(N):
            pts = self.bisect(ds[i], x)
            x = pts[0]
            pts_hist.append(pts)
        pts_hist = np.array(pts_hist)
        return pts_hist
    
    def log_info(self, msg):
        if self.tdalogger is not None:
            self.tdalogger.info(msg)
        
    def log_warn(self, msg):
        if self.tdalogger is not None:
            self.tdalogger.warning(msg)
        
    def run(self, data, N , ds, dt):
        # print(start_x)
        quality_status = self.test_waypoints(data) 
        ## test for variance after fitting to curve
        # self.lane_model(data[:,0])
        # var, Y, Y_pred = self.calculate_variance(data)
        if self.verbose:
            print ("quality_status", quality_status)
        try:
            if quality_status == True:
                speed = self.len_points/dt
                # print(speed)
                pts_hist = self.get_N_points(start_x = 0.0, ds = ds * speed, N = N) 
                return pts_hist       
        except:
            raise NameError("Quality check of waypoint is not passed") 
            
def main():
    w_corr = ParametricWaypoint()       
    
    ############# TEST 1 ###############################################
    a=3; b=5; x0=7; ds = 1234
    w_corr.param =np.array([a,b,0])    
    pts = w_corr.bisect(ds, x0)    
    ## the value should be [20.907551885003034, 1415.914936897297]
    
    ####################################################################
    
    ############# TEST 2 ###############################################
    # Ground truth curve
    a = 0.02
    b = 0.2
    c = 2
    
    # generate data for testing with random noise and ununiform x distribution
    x_start = 0
    x_end = 15
    std = 1.5

    # print (np.random.set_state)
#     np.random.seed(20)
    x_random = np.random.uniform(x_start, x_end,20)
    noise = np.random.normal(0,np.ones(len(x_random))*std)
    # print ("Seed state for debugging error", np.random.get_state())

    x_random += noise
    x = np.sort(x_random) # remove sorting to test the
    noise = np.random.normal(0,np.ones(len(x_random))*std*0.3)
    Y = np.array(a*x**2 + b*x + c*np.ones(len(x))) + noise
    data_new = np.vstack([x, Y]).T
    t0 = time.time()
    pts_hist = w_corr.run(data_new)
    var, _, Y_pred = w_corr.calculate_variance(data_new)
    print ("time taken ", time.time() - t0)
    
    ####################################################################
    
    #################### STATS #########################################
    x_pred, y_pred = pts_hist[:,0], pts_hist[:,1]
    dist = np.sqrt((x_pred[1:] - x_pred[:-1])**2 + (y_pred[1:] - y_pred[:-1])**2)
    print ("Stats of equidstant", dist)
    print ("Ground Truth curve parameter", [a, b, c])
    print ("Obtained curve parameter", w_corr.param)


    ####################################################################
    
    #################### PLOT ##########################################
    fig, axs = plt.subplots(1, 1, figsize=(16,10))
    axs.plot(x, Y, '--o', color = 'red', label = r'Waypoints',linewidth=4, alpha = 0.6)
    # axs.plot(x, Y_pred, '--o', color = 'goldenrod', label = r'Predicted Points',linewidth=2, alpha = 0.8, markersize = 10)
    axs.plot(x_pred, y_pred, '-o', color = 'darkblue', label = r'Predicted Feasible Equidistance Points',linewidth=2, alpha = 0.4)
    axs.grid()
    axs.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3, fontsize=15, scatterpoints = 1)
    axs.set_xlabel(r"X-axis [meter]", fontsize=15)
    axs.set_ylabel(r"Y-axis [meter]", fontsize=15)
    axs.set_title('Actual vs predicted curve coordinates', fontsize=15, fontweight='bold', color='#30302f', loc='center', y=1.1)
    plt.show()

if __name__ == '__main__':
    main() 
    
    

####################### MISC FUNCTIONS (NOT USED) ############################

def equidistance_points(self, mpc_ds, model_coeff, N):
    
    x = 0.0
    s = mpc_ds

    a, b, c = model_coeff[:3]
    # a, b_l, b_r, c_l, c_r, width = model_coeff
    # b = (b_l + b_r)/2
    # c = (c_l+c_r)/2

    path_x = []
    path_y = []
    for i in range(N):
        path_x.append(x)
        y = a*x**2 + b*x + c
        path_y.append(y)
        if abs(a) < 0.0001 and False:             
            x = ((3*a*s + (b + 2*a*x + 1)**(3/2))**(2/3) - (1+b))/(2*a)
        else: 
            # x = s/np.sqrt(1+b) + x
            x = np.sqrt(s**2 - b**2)/2 + x
        x = x.real if not isinstance(x, float) else x
    return np.array(path_x) + 0.0, np.array(path_y)


def interpcurve(self,N,pX,pY):
    #equally spaced in arclength
    N=np.transpose(np.linspace(0,1,N))

    #how many points will be uniformly interpolated?
    nt=N.size

    #number of points on the curve
    n=pX.size
    pxy=np.array((pX,pY)).T
    p1=pxy[0,:]
    pend=pxy[-1,:]
    last_segment= np.linalg.norm(np.subtract(p1,pend))
    epsilon= 10*np.finfo(float).eps

    #IF the two end points are not close enough lets close the curve
    if last_segment > epsilon*np.linalg.norm(np.amax(abs(pxy),axis=0)):
        pxy=np.vstack((pxy,p1))
        nt = nt + 1
    else:
        print('Contour already closed')

    pt=np.zeros((nt,2))

    #Compute the chordal arclength of each segment.
    chordlen = (np.sum(np.diff(pxy,axis=0)**2,axis=1))**(1/2)
    #Normalize the arclengths to a unit total
    chordlen = chordlen/np.sum(chordlen)
    #cumulative arclength
    cumarc = np.append(0,np.cumsum(chordlen))

    tbins= np.digitize(N,cumarc) # bin index in which each N is in

    #catch any problems at the ends
    tbins[np.where(tbins<=0 | (N<=0))]=1
    tbins[np.where(tbins >= n | (N >= 1))] = n - 1      

    s = np.divide((N - cumarc[tbins]),chordlen[tbins-1])
    pt = pxy[tbins,:] + np.multiply((pxy[tbins,:] - pxy[tbins-1,:]),(np.vstack([s]*2)).T)

    return pt 