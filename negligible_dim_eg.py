"""
This file is used to provide an example of how we can ignore the 
very small dimension of a data in LID calculation. E.g. a pancake
has 3 dim of (height, width, depth); however, the depth is very small
= very small variance, hence, ignorable. Or, when a gaussian has different
(very small) standard deviation (variance) in one dimension compared to others
when calculating the LID, we can ignore that one. Based on the LIDL paper, if
the (added noise's STD > 10*STD of a dimension (usually the small one)), the LIDL
calculation will ignore that dimension and won't count it as another dimension. 
"""

from generate_data import *

# Generate 1000 samples from a 10D Gaussian
n_samples = 1000
dim = 10
mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
std = [1.0, 0.5, 2.06, 0.2, 1.3, 1.4, 1.6, 0.7, 2.3, 2.5]

samples = generate_gaussian(n_samples, dim, mean, std)
print(samples.shape) 