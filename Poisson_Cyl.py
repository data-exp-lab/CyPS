import numpy as np
import scipy.integrate as integrate
import nufft3df90

class Poisson_Cyl:
    """
    A Poisson solver for a sphere of uniform density rho and radius R0 sitting at (x0, y0, z0)
    (x0, y0, z0) are coordinates in Cartesian coordinate 
    NN is the resolution in r-direction
    method can be convolution or nufft
    """
    
    def __init__(self, NN, method="convolution"):

        self.method = method

        if (self.method != "convolution" and self.method != "nufft"):
            print("Method has to be either convolution or nufft. Reset method to convolution.")
            self.method = "convolution"
            
        # simulation domain
        self._x1_min = 0.2
        self._x1_max = 8.2
        self._x2_min = 0.0
        self._x2_max = 2.0*np.pi
        self._x3_min = 0.0
        self._x3_max = 8.0
        
        self._N1 = NN
        self._N2 = 8*NN
        self._N3 = NN
        
        # init simulation grid and parameters
        self.__initialize()
        
        
    def __initialize(self):
        
        # private vars
        self.__grid_shape = (self._N1, self._N2, self._N3)
        
        self.__x1_range = self._x1_max-self._x1_min
        self.__x2_range = self._x2_max-self._x2_min
        self.__x3_range = self._x3_max-self._x3_min
        
        self.__dx1   = (self.__x1_range)/float(self._N1)
        self.__dx2   = (self.__x2_range)/float(self._N2)
        self.__dx3   = (self.__x3_range)/float(self._N3)
        
        self.__x1, self.__x2, self.__x3 = np.mgrid[self._x1_min+0.5*self.__dx1: self._x1_max-0.5*self.__dx1: self._N1*1j, 
                                                   self._x2_min+0.5*self.__dx2: self._x2_max-0.5*self.__dx2: self._N2*1j, 
                                                   self._x3_min+0.5*self.__dx3: self._x3_max-0.5*self.__dx3: self._N3*1j]
        
        ## cartesian coordinate
        self.__crt_x1 = self.__x1*np.sin(self.__x2)
        self.__crt_x2 = self.__x1*np.cos(self.__x2)
        self.__crt_x3 = self.__x3
        
        ## 
        self.__rho_array = np.zeros(self.__grid_shape)
        #self.__dist2center = np.sqrt( (self.__crt_x1-self.x0)**2.0 + (self.__crt_x2-self.y0)**2.0 + (self.__crt_x3-self.z0)**2.0 )
        
        ## save result
        self.__solution = np.full(self.__grid_shape, None)
        
        ##
        if (self.method == "nufft"):
            self.__ms  = 2*self._N1
            self.__mt  = 2*self._N2
            self.__mu  = 2*self._N3        
            self.__eps = 10.0**(-5.0)
            
        
        ## init method specific coord 
        if (self.method == "convolution"):
            self.__init_convolution()
        elif (self.method=="nufft"):
            self.__init_nufft()
        
        
    def init_problem(self, problem):
        self.problem = problem
        self.__initialize()
        self.__rho_array = self.problem.get_rho()
        
    
    
    def __init_convolution(self):
        self.__kernel_out = np.full((self._N1, self._N1, self._N2, 2*self._N3), None)
    
    
    def __init_nufft(self):
        # nufft parameters
        self.__volume = float(self.__ms)*float(self.__mt)*float(self.__mu)
        
        # coordinate related
        self.__crt_x1_min = -1.0 * self._x1_max
        self.__crt_x1_max =  1.0 * self._x1_max
        self.__crt_x2_min = -1.0 * self._x1_max
        self.__crt_x2_max =  1.0 * self._x1_max
        self.__crt_x3_min =  self._x3_min
        self.__crt_x3_max =  self._x3_max
        
        self.__crt_x1_range = self._x1_max * 2.0
        self.__crt_x2_range = self._x1_max * 2.0
        self.__crt_x3_range = self.__x3_range
        
        self.__crt_dx1 = self.__crt_x1_range / float(self._N1)
        self.__crt_dx2 = self.__crt_x2_range / float(self._N2)
        self.__crt_dx3 = self.__crt_x3_range / float(self._N3)
        
        # create a uniform cartesian grid
        self.__crt_x1_L, self.__crt_x2_L, self.__crt_x3_L = \
         np.mgrid[self.__crt_x1_min+0.5*self.__crt_dx1: self.__crt_x1_max+self.__crt_x1_range-0.5*self.__crt_dx1: (2*self._N1)*1j, 
                  self.__crt_x2_min+0.5*self.__crt_dx2: self.__crt_x2_max+self.__crt_x2_range-0.5*self.__crt_dx2: (2*self._N2)*1j, 
                  self.__crt_x3_min+0.5*self.__crt_dx3: self.__crt_x3_max+self.__crt_x3_range-0.5*self.__crt_dx3: (2*self._N3)*1j]
                  
        # replace the lower left corner w/ non-uniform grid 
        self.__crt_x1_L[0:self._N1, 0:self._N2, 0:self._N3] = self.__crt_x1
        self.__crt_x2_L[0:self._N1, 0:self._N2, 0:self._N3] = self.__crt_x2
        self.__crt_x3_L[0:self._N1, 0:self._N2, 0:self._N3] = self.__crt_x3
        
        # save 
        self.__green = np.full((2*self._N1, 2*self._N2, 2*self._N3), None)
        self.__green_k = np.full((self.__ms, self.__mt, self.__mu), None)
        
        
    def reset_nufft_parameter(self, ms, mt, mu, eps):
        self.__ms  = ms
        self.__mt  = mt
        self.__mu  = mu
        self.__eps = eps
        
        self.__init_nufft()

        
    def __kernel(self):
        """ this is the Kernel function w/o r-correction """
        if (self.__kernel_out.all() != None):
            return self.__kernel_out
        
        r, r_p, theta, z = np.mgrid[(self._x1_min+0.5*self.__dx1): (self._x1_max-0.5*self.__dx1): self._N1*1j,
                                    (self._x1_min+0.5*self.__dx1): (self._x1_max-0.5*self.__dx1): self._N1*1j,
                                    0.0:(self.__x2_range-self.__dx2):self._N2*1j, 
                                    0.0:(self.__x3_range-self.__dx3):self._N3*1j]
                                    
        kernel_out = np.zeros((self._N1, self._N1, self._N2, 2*self._N3))
        kernel_sub = np.zeros((self._N1, self._N1, self._N2, self._N3))
        
        denominator = np.sqrt( (r-r_p)**2.0 + 2.0*r*r_p*(1.0-np.cos(theta)) + z**2.0 )
        idx = denominator != 0.0
        kernel_sub[idx] = - self.__dx2*self.__dx3 / denominator[idx]
        
        kernel_flip = np.flip(kernel_sub, 3)
        kernel_out[:, :, :, 0:self._N3]            = kernel_sub
        kernel_out[:, :, :, self._N3+1:2*self._N3] = kernel_flip[:, :, :, 0:self._N3-1]
        
        # fix kernel_out[:, :, :, N3]
        for i in range(self._N1):
            for j in range(self._N1):
                for k in range(self._N2):
                    kernel_out[i,j,k,self._N3] = np.sqrt( (r[i,j,k,0]-r_p[i,j,k,0])**2.0 
                                                           +2.0*r[i,j,k,0]*r_p[i,j,k,0]*(1.0-np.cos(theta[i,j,k,0])) 
                                                           +self._x3_max**2.0 )
        
        kernel_out *= self.__dx1
        self.__kernel_out = kernel_out
        
        return kernel_out
    
    
    def get_crt_coord(self):
        return (self.__crt_x1, self.__crt_x2, self.__crt_x3)
    
    
    def __prepare_convolution_rho(self):
        rho_L = np.zeros((self._N1, self._N2, 2*self._N3))
        rho_L[0:self._N1, 0:self._N2, 0:self._N3] = self.__rho_array*self.__x1
        
        return rho_L
    
    
    def __convolution(self):
        if (self.__solution.all() != None):
            return self.__solution
            
        rho = self.__prepare_convolution_rho()
        
        if (self.__kernel_out.any() == None):
            kernel = self.__kernel()
        else:
            kernel = self.__kernel_out
        
        rho_k = np.fft.fft(rho, axis=2)
        rho_k = np.fft.fft(rho_k, axis=1) 
        
        kernel_k = np.fft.fft(kernel, axis=3)
        kernel_k = np.fft.fft(kernel_k, axis=2)
        
        Phi_k = np.zeros((self._N1, self._N2, 2*self._N3), dtype=complex)
        ### -- caution: this is superb slow in python -- ###
        for i in range(self._N1):
            for j in range(self._N1):
                for k in range(self._N2):
                    for l in range(2*self._N3):
                        Phi_k[j,k,l] += rho_k[i,k,l] * kernel_k[i,j,k,l]
        
        Phi_L = np.fft.ifft(Phi_k, axis=2)
        Phi_L = np.fft.ifft(Phi_L, axis=1)
        
        Phi_out = Phi_L[0:self._N1, 0:self._N2, 0:self._N3]
        Phi_out = Phi_out.real
        
        return Phi_out
        
    
    def __init_green(self):
        if (self.__green.all() != None):
            return self.__green
        
        green = np.zeros((2*self._N1, 2*self._N2, 2*self._N3))
        
        for i in range(2*self._N1):
            for j in range(2*self._N2):
                for k in range(2*self._N3):
                    if (k>self._N3):
                        kk = float(2*self._N3-k)
                    else:
                        kk = float(k)
                
                    if (j>self._N2):
                        jj = float(2*self._N2-j)
                    else:
                        jj = float(j)
            
                    if (i>self._N1):
                        ii = float(2*self._N1-i)
                    else:
                        ii = float(i)
            
                    denominator = np.sqrt((ii*self.__crt_dx1)**2.0+(jj*self.__crt_dx2)**2.0+(kk*self.__crt_dx3)**2.0)
                    
                    if (denominator != 0.0):
                        green[i, j, k] = -self.__crt_dx1*self.__crt_dx2*self.__crt_dx3 / denominator

        # fix green[0,0,0]       
        green[0,0,0] = - self.__crt_dx1*self.__crt_dx2*self.__crt_dx3/np.min([self.__crt_dx1,self.__crt_dx2,self.__crt_dx3])
        #green[0,0,0] = 0.0
        
        self.__green = green
        return self.__green
    
    
    def __prepare_green_k(self):
        if (self.__green_k.all() != None):
            return self.__green_k
            
        x1_L, x2_L, x3_L = np.mgrid[0.0: 2.0*np.pi-(2.0*np.pi)/(2.0*self._N1): (2*self._N1)*1j,
                                    0.0: 2.0*np.pi-(2.0*np.pi)/(2.0*self._N2): (2*self._N2)*1j,
                                    0.0: 2.0*np.pi-(2.0*np.pi)/(2.0*self._N3): (2*self._N3)*1j]
    
        x1_array = np.array(x1_L.ravel(), dtype=np.float64)
        x2_array = np.array(x2_L.ravel(), dtype=np.float64)
        x3_array = np.array(x3_L.ravel(), dtype=np.float64)
    
        green = self.__init_green()
        green_flat = np.array(green.ravel(), dtype=np.complex128)
    
        green_k, ier = nufft3df90.nufft3d1f90(x1_array, x2_array, x3_array, green_flat, 1, self.__eps, self.__ms, self.__mt, self.__mu)
        green_k *= self.__volume
    
        if (ier != 0):
            print("NUFFT failed in " + __green_k.__name__ )
        
        self.__green_k = green_k
    
        return self.__green_k
        
        
    def __prepare_nufft_rho(self):
        rho_L = np.zeros((2*self._N1, 2*self._N2, 2*self._N3))
        
        rho_L[0:self._N1, 0:self._N2, 0:self._N3] = \
            self.__rho_array * (self.__x1*self.__dx1*self.__dx2) / (self.__crt_dx1*self.__crt_dx2)
                
        return rho_L
        
        
    def __prepare_nufft_grid(self):
        x1_L = ( self.__crt_x1_L-(self.__crt_x1_min+0.5*self.__crt_dx1) ) / (2.0*self.__crt_x1_range) * (2.0*np.pi)
        x2_L = ( self.__crt_x2_L-(self.__crt_x2_min+0.5*self.__crt_dx2) ) / (2.0*self.__crt_x2_range) * (2.0*np.pi)
        x3_L = ( self.__crt_x3_L-(self.__crt_x3_min+0.5*self.__crt_dx3) ) / (2.0*self.__crt_x3_range) * (2.0*np.pi)
    
        x1_flat = np.array(x1_L.ravel(), dtype=np.float64)
        x2_flat = np.array(x2_L.ravel(), dtype=np.float64)
        x3_flat = np.array(x3_L.ravel(), dtype=np.float64)
        
        return (x1_flat, x2_flat, x3_flat)
    
    
    def __nufft_rho_k(self):
        rho  = self.__prepare_nufft_rho()
        x1, x2, x3 = self.__prepare_nufft_grid()
        
        rho_flat = np.array(rho.ravel(), dtype=np.complex128)
        rho_k, ier = nufft3df90.nufft3d1f90(x1, x2, x3, rho_flat, 1, self.__eps, self.__ms, self.__mt, self.__mu)
        rho_k *= self.__volume
        
        if (ier != 0):
            print("NUFFT failed in " + __nufft_rho_k.__name__ )
            
        return rho_k
        
        
    def __nufft(self):
        if (self.__solution.all() != None):
            return self.__solution
            
        green_k = self.__prepare_green_k()
        rho_k   = self.__nufft_rho_k()
        
        Phi_k = green_k * rho_k
        
        x1, x2, x3 = self.__prepare_nufft_grid()
        
        Phi_L, ier = nufft3df90.nufft3d2f90(x1, x2, x3, -1, self.__eps, self.__ms, self.__mt, self.__mu, Phi_k)
        Phi_L /= self.__volume
        Phi_L = np.reshape(Phi_L, (2*self._N1, 2*self._N2, 2*self._N3))
        
        if (ier != 0):
            print("NUFFT failed in " + __nufft.__name__ )
        
        Phi_out = np.zeros(self.__grid_shape)
        Phi_out = Phi_L[0:self._N1, 0:self._N2, 0:self._N3]
        Phi_out = Phi_out.real
        
        return Phi_out
    
    
    def solve(self):
        """ solve for potential and save the solution  """
        if (self.method == "convolution"):
            if (self.__solution.any() == None):
                self.__solution = self.__convolution()
                
            return self.__solution
            
        elif (self.method == "nufft" ):
            if (self.__solution.any() == None):
                self.__solution = self.__nufft()
                
            return self.__solution
    
    
    def analytic_sol(self):
        return self.problem.get_analytic()
    
    
    def error(self):
        """ error \equiv solution - analytic """
        if self.__solution.any() == None:
            self.solve()
            
        return self.__solution - self.analytic_sol()
    
    
    def L1_error(self):
        """ output L1 error """
        error = self.error()
        analytic_sol = self.analytic_sol()
        
        L1_error = np.sum( np.abs(error) ) / np.sum(np.abs(analytic_sol))
        
        return L1_error




#############
#############

class Problem():
    def __init__(self, x_grid, y_grid, z_grid, problem):
        """
        problem: sphere, spheroid, user
        grid   : underlying Carteisan grid
        """
        
        self.problem = problem
        if (problem != "sphere" and problem != "spheroid" and problem != "user"):
            print("problem has to be sphere, spheroid or user. Reset problem to user.")
            self.problem = "user"
            
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._z_grid = z_grid
        self.__grid_shape = x_grid.shape
        self._analytic   = np.zeros(self.__grid_shape, dtype=np.float32)
        self._rho_array  = np.zeros(self.__grid_shape, dtype=np.float32)
    
    
    def get_rho(self):
        pass
        
    def get_analytic(self):
        print("No existing implemented analytic solution.")
        


class Sphere(Problem):
    def __init__(self, rho, R0, x0, y0, z0, x_grid, y_grid, z_grid):
        """
        rho:          density of the sphere
        R0:           radius of the sphere
        (x0, y0, z0): center of the sphere
        """
        problem = "sphere"
        super().__init__(x_grid, y_grid, z_grid, problem)
        self.add_sphere(rho, R0, x0, y0, z0)
        
        
    def __analytic_sol(self, rho, R0):
        rho = float(rho)
        R0  = float(R0)
        
        idx = self.__dist2center <= R0
        Phi_analytic = -(4.0/3.0)*np.pi*rho * R0**3.0 / self.__dist2center
        
        dist_array = self.__dist2center[idx]
        Phi_analytic[idx] = -(2.0/3.0)*np.pi*rho*( 3.0*R0**2.0 - dist_array**2.0 )
        
        return Phi_analytic
        
            
    def add_sphere(self, rho, R0, x0, y0, z0):
        self.__dist2center = \
            np.sqrt( (self._x_grid-x0)**2.0 + (self._y_grid-y0)**2.0 + (self._z_grid-z0)**2.0 )
            
        idx = self.__dist2center <= R0
        self._rho_array[idx] += rho
        self._analytic       += self.__analytic_sol(rho, R0)
        
        
    def get_analytic(self):
        return self._analytic
        
    def get_rho(self):
        return self._rho_array
        
        

class Spheroid(Problem):
    def __init__(self):
        pass
    def _analytic_sol(self):
        pass
            


class User(Problem):
    def __init__(self, rho_array, x_grid, y_grid, z_grid, problem="user"):
        super().__init__(problem, x_grid, y_grid, z_grid)
        
    def get_rho():
        return rho_array
