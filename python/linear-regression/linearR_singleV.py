from numpy import loadtxt,zeros,ones,array,linspace,logspace
from pylab import scatter,show,title,xlabel,ylabel,contour,plot

def compute_cost(x,y,theta):
	m=y.size
	predictions=x.dot(theta).flatten()
	sqerrors=(predictions-y) ** 2
	j = (1.0 / (2 * m)) * sqerrors.sum()
	return j

def gradient_descent(x,y,theta,alpha,num_iters):

	m=y.size
	j_history=zeros(shape=(num_iters,1))

	for i in range(num_iters):
		predictions=x.dot(theta).flatten()
		errors_x1=(predictions-y)*x[:,0]
		errors_x2=(predictions-y)*x[:,1]

		theta[0][0]=theta[0][0]-alpha*(1.0/m)*errors_x1.sum()
		theta[1][0]=theta[1][0]-alpha*(1.0/m)*errors_x2.sum()

		j_history[i,0]=compute_cost(x, y, theta)

	return theta,j_history


if __name__=='__main__':
	# load dataset

    data=loadtxt('ex1data1.txt',delimiter=',')

    #plot the data

    scatter(data[:,0],data[:,1],marker='o',c='b')
    title('profit distribution')
    xlabel('population of city in 10,000')
    ylabel('profit in 10,000')
    #show()
    x=data[:,0]
    y=data[:,1]
    m=y.size
    it=ones(shape=(m,2))
    it[:,1]=x
    theta=zeros(shape=(2,1))
    num_iters=1500
    alpha=0.01
    # initial cost
    initial_cost=compute_cost(it, y, theta)
    print "initial_cost is : "
    print initial_cost
    #theta an jhistory
    theta,j_history=gradient_descent(it, y, theta, alpha, num_iters)
    print theta
    print j_history
    # predict values
    predict1=array([1,3.5]).dot(theta).flatten()
    print 'For population = 35,000, we predict a profit of %f' % (predict1 * 10000)

    #plot results

    result=it.dot(theta).flatten()
    plot(data[:,0],result)
    show()
    #grid  to calculate j

    theta0_vals=linspace(-10,10,100)
    theta1_vals=linspace(-1,4,100)

    j_vals=zeros(shape=(theta0_vals.size,theta1_vals.size))
    # fil  j values
    for p, q in enumerate(theta0_vals):
    	for r,s in enumerate(theta1_vals):
    		thetaT=zeros(shape=(2,1))
    		thetaT[0][0]=q
    		thetaT[1][0]=s
    		j_vals[p][r]=compute_cost(it, y, thetaT)

    #contour plot
    j_vals=j_vals.T
    #plot j_vals as 15 contour plots
    contour(theta0_vals,theta1_vals,j_vals,logspace(-2,3,20))
    xlabel('theta0_vals')
    ylabel('theta0_vals')
    scatter(theta[0][0],theta[1][0])
    show()











