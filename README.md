# CyPS: cyindrical poisson solver 
Solving Poisson equation $\nabla^2 \Phi = 4 \pi G \rho$ under a cylindrical grid.

## Method
### 1) Convolution
$$
\mathbf{
\tilde{\Phi} (r, k_\phi, k_z) = 
G \int^\infty_{0} dr' \; \tilde{K}(r, r', k_\phi, k_z) \; \tilde{\rho}(r', k_\phi, k_z)
}
$$

### 2) NUFFT
Use NUFFT package from CMCL at NYU: https://cims.nyu.edu/cmcl/nufft/nufft.html
