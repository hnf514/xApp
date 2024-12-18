import sys
import matplotlib.pyplot as plt 
import gzip, os
import numpy as np
from scipy.stats import multivariate_normal

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# Function that downloads a specified MNIST data file from Yann Le Cun's website
def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

# Invokes download() if necessary, then reads in images
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    print (data.shape)
    print (data[8888])

    return data

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        print (data.shape)

    return data

## Load the training set
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')

## Load the testing set
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
def displaychar(image):
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
displaychar(train_data[58])


print ("train_data.shape",train_data.shape)
print("train_labels.shape",train_labels.shape)


def fit_generative_model(x,y):
    k = 10  # labels 0,1,...,k-1
    d = (x.shape)[1]  # number of features
    mu = np.zeros((k,d))
    sigma = np.zeros((k,d,d))
    pi = np.zeros(k)
    ###
    for label in range(k):
        class_data = x[y == label]
        
        # Estimate mean
        mu[label] = np.mean(class_data, axis=0)
        
        # Estimate covariance matrix
        sigma[label] = np.cov(class_data, rowvar=False)
        
        # Estimate frequency
        pi[label] = len(class_data) / len(y)
        print ("mu shape:",mu.shape)
        print ("sigma shape:",sigma.shape)

        print ("pi shape:",pi.shape)
    return mu, sigma, pi    ###

    # Halt and return parameters



mu, sigma, pi = fit_generative_model(train_data, train_labels)
displaychar(mu[0])
displaychar(mu[1])
displaychar(mu[2])

print ("sigma[1] :",sigma[1])




# Compute log Pr(label|image) for each [test image,label] pair.
k = 4
score = np.zeros((len(test_labels),k))
label=1
covariance = np.array([[0.5, 0, 0],
                       [0, 0.9, 0],
                       [0, 0, 1]])
# rv = multivariate_normal(mean=mu[label], cov=sigma[label])
rv = multivariate_normal(mean=[1, 5, 4], cov=covariance)
sample = rv.rvs(199)
print("Generated Sample:")
print(sample)
