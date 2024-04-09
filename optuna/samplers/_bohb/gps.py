import numpy as np
import gpytorch
import torch
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import minmax_scale
import GPy



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=None, kernel=None, training_iter=50, normalize=True):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.normalization=normalize
        if normalize: 
            train_x = self.minmax_scale(train_x)

        self.scaling_factor = 0.0
        if self.scaling_factor>=0:
            train_y = deepcopy(train_y/self.scaling_factor)

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if kernel is None:
            #self.kernel = gpytorch.kernels.RBFKernel()
            self.kernel = gpytorch.kernels.MaternKernel()
            

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel)
        self.training_iter=training_iter
        self.print = False
        self.train_x, self.train_y = train_x, train_y


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def train_gp(self, X=None, y=None):
        # Find optimal model hyperparameters

        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()

            if self.print:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, self.training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()
                ))

            optimizer.step()


    def eval_gp(self, test_x):

        if self.normalization: 
            test_x = self.minmax_scale(test_x)

        self.eval()
        self.likelihood.eval()

        f_preds = self(test_x)
        #y_preds = self.likelihood(self(test_x))

        f_mean = f_preds.mean
        f_var = f_preds.variance
        #f_covar = f_preds.covariance_matrix
        #f_samples = f_preds.sample(sample_shape=torch.Size(1000,))

        #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #    observed_pred = self.likelihood(self(test_x))

        #ret = [(f_mean[i].item(),f_var[i].item()) for i in range(len(test_x))]
        
        if self.scaling_factor<=0: 
            return f_mean, f_var

        return f_mean*self.scaling_factor, f_var*self.scaling_factor
    

    def minmax_scale(data, feature_range=(0, 1)):
        min_val, max_val = feature_range
        scaled_data = (data - data.min()) / (data.max() - data.min()) * (max_val - min_val) + min_val
        return scaled_data



class EasyGPModel1():
    def __init__(self, normalize=True):
        #kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e2))
        kernel = ConstantKernel(1.0, (1e-7, 1e7)) * Matern(length_scale=1.0, nu=2.5)

        #kernel = 1.0 * Matern(length_scale=100.0, length_scale_bounds=(1e-7, 1e7), nu=2.5)
        #kernel.k2__length_scale = (0.1, 1e5)  # Adjust the upper bound

        # Create the GaussianProcessRegressor
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=True)
        self.scaling_factor = 0.0
        
        self.normalization=normalize


    def train_gp(self, X, y):
        X_norm = minmax_scale(X, axis=0) if self.normalization else X
        y_scale = y/self.scaling_factor if self.scaling_factor>0 else y

        self.model.fit(X_norm, y_scale)


    def eval_gp(self, X_test):
        X_norm =  minmax_scale(X_test, axis=0) if self.normalization else X_test
        y_pred, sigma = self.model.predict(X_norm, return_std=True)

        if self.scaling_factor<=0: 
            return y_pred, sigma

        return y_pred*self.scaling_factor, sigma*self.scaling_factor
    

class EasyGPModel():
    def __init__(self, normalize=True):
        # Define the Matern kernel
        return

    def train_gp(self, X, y):
        self.kernel = GPy.kern.Matern52(input_dim=X.shape[1], variance=1.0, lengthscale=1.0)

        self.gp = GPy.models.GPRegression(X, y[:, None], self.kernel)
        self.gp.optimize(messages=False)


    def eval_gp(self, X_test):
        y_pred, sigma = self.gp.predict(X_test)
        return y_pred, sigma
    
