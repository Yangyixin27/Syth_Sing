
import numpy as np

# model type 
HARMONIC = 1
APERIODIC = 0
PITCH = 0

# sigmoid function
def sigmoid(x):
	y = 1/(1+np.exp(-x))
	return y

if HARMONIC: 
	# acoustic feature dimension
	feat_dim = 60 

	# temperature 
	x = np.linspace(1,4,4)
	b = 0.05 
	slope = (0.5-0.05) / 5
	pts = np.zeros(4)
	for i in range(0,4):
		pts[i] = slope*x[i] + b
	print(pts)

	tau = np.append(0.05*np.ones(3),pts)
	tau = np. append(tau, 0.5*np.ones(60-7))
	print(tau)


if APERIODIC:
	# acoustic feature dimension
	feat_dim = 4
	# temperature 
	tau = 0.01

if PITCH:
	# acoustic feature dimension
	feat_dim = 1
	# temperature
	tau = 0.01

#################### TO DO: REPLACE WITH A0-A3 FOUND FROM FRAME ##################
num_params = 4
frame = np.random.rand(feat_dim, num_params)

# get a0-a3
a0 = frame[:,0]
a1 = frame[:,1]
a2 = frame[:,2]
a3 = frame[:,3]

# testing 
# print(frame.shape)
# print(a0.shape)
# print(a1.shape)
# print(a2.shape)
# print(a3.shape)

# apply nonlinearities to obtain free parameters in suitable ranges 
location = 2*sigmoid(a0)-1
scale =  2/255 * np.exp(4*sigmoid(a1))
skewness = 2*sigmoid(a2) - 1 
shape = 2*sigmoid(a3)

# testing 
# print(location.shape)
# print(scale.shape)
# print(skewness.shape)
# print(shape.shape)

# constants tuned by hand 
gamma_u = 1.6
gamma_s = 1.1
gamma_w = 1/1.75

# map predicted location, scale, skewness, and shape to Gaussian mixture parameters 
num_gauss = 4
sigma = np.zeros((num_gauss,feat_dim))
omega = np.zeros((num_gauss,feat_dim))
mu = np.zeros((num_gauss,feat_dim))

for k in range(0,num_gauss):
	sigma[k] = skewness * np.exp((abs(shape)*gamma_s-1)*k)
	sum = 0 
	for i in range(0,num_gauss):
		sum = sum + skewness**(2*i) * shape**i * gamma_w**i 
	omega[k] = (skewness**(2*k) * shape**k * gamma_w**k) / sum

for k in range(0,num_gauss):
	sum = 0
	for i in range(0,k):
		sum = sum +  omega[k] * gamma_u * skewness 
	mu[k] = location + sum 

# testing 
# print(sigma.shape)
# print(mu.shape)
# print(omega.shape)

# calc global weighted average 
mu_gwa = np.zeros(feat_dim)
for r in range(0,feat_dim):
	mu_gwa[r] = np.sum(sigma[:,r]*omega[:,r])

# testing 
# print(mu_gwa.shape)

# shift component means towards global weighted average 
mu_shifted = np.zeros((num_gauss, feat_dim))
for k in range(0,num_gauss):
	mu_shifted[k] = mu[k] + (mu_gwa - mu[k]) * (1 - tau)

# scale variance components by temperature
sigma_scaled= np.zeros((num_gauss, feat_dim))
for k in range(0,num_gauss):
	sigma_scaled[k] = sigma[k] * np.sqrt(tau)	

# testing 
# print(mu_shifted.shape)
# print(sigma_scaled.shape)
