from numpy import loadtxt,zeros,ones,array,linspace,logspace,mean,std
from pylab import scatter,show,title,xlabel,ylabel,contour,plot

def feature_normalize(x):
	mean_r=[]
	std_r=[]
	x_normalize=x
	m=x.shape[1]
	for i in range(m):
		m=mean(x[:,i])
		s=std(x[:,i])
		mean_r.append(m)
		std_r.append(s)
		x_normalize[:,i]=(x_normalize[:,i]-m)/s
	return x_normalize,mean_r,std_r

def compute_cost(x,y,theta):
	m=y.size
	predictions=x.dot(theta)
	sqerrors=(predictions-y)
	j = (1.0 / (2 * m)) * sqerrors.T.dot(sqerrors)
	return j

def gradient_descent(x,y,theta,alpha,num_iters):
	m=y.size
	j_history=zeros(shape=(num_iters,1))
	for i in range(num_iters):
		predictions=x.dot(theta)
		theta_size=theta.size
		for it in range(theta_size):
			temp=x[:,it]
			temp.shape=(m,1)
			errors_x1=(predictions-y)*temp
			theta[it,0]=theta[it,0]-alpha*(1.0/m)*errors_x1.sum()
		j_history[i,0]=compute_cost(x, y, theta)
	return theta,j_history
if __name__=='__main__':
	data = loadtxt('ex1data2.txt', delimiter=',')
	X=data[:,(1,2)]
	y=data[:,2]
	m=y.size
	y.shape=(m,1)
	x,mean_r,std_r=feature_normalize(X)
	it=ones(shape=(m,3))
	it[:,1:3]=x
	theta=zeros(shape=(2,1))
	iterations=100
	alpha=0.01
	theta,j_history=gradient_descent(it, y, theta, alpha, iterations)
	print theta,j_history




