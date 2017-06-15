#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
'''
1.1 线性模型
$z=f(w,x)=XW+b=w_1x_1+w_2x_2+...+w_px_p$
x是输入，w是参数,z是模型的输出， 线性回归调整模型参数使输出z与实际标签y的误差最小化。
模型中的参数w存放在coef_中, b在intercept_中。
'''



'''
1.1.1 一般最小二乘法
平均平方误差：$min||Xw-y||_2^2$。
时间复杂度：使用奇异值分解来解决最小二乘问题，如果X是(n,p)的矩阵,该方法的时间复杂度是N(np^2),n>=p
'''
#线性回归模型
reg=linear_model.LinearRegression()
x=np.array(range(10))
x=np.reshape(x,[10,1])
y=x*2+np.random.normal(loc=0.0,scale=1.0,size=10)+10
#训练
reg.fit(X=x,y=y)
#输入模型的参数
print(reg.coef_)
print(reg.intercept_)


'''
1.1.2 岭回归 Ridge
(1) 岭回归通过使用L2正则项对回归模型中的参数进行惩罚，
(2) 其中使用alpha参数调整惩罚项的大小,alpha越大对参数的惩罚越大（the greater the amount of shrinkage），模型越robust。
(3) 可以使用RidgeCV 从多个alpha中选择一个合适的，RidgeCV在内部实现了交叉验证。
'''
# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# RidgeCV使用交叉验证选择合适的alpha
reg=linear_model.RidgeCV(alphas=[1e-8,1e-5,1e-4,0.01,0.1,1.0,10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.alpha_)

'''
1.1.3 Lasso
（1）. Lasso是可以估计稀疏参数的线性模型, 当只需要依赖很少的特征时，Lasso很有用，因此Lasso以及其变种是压缩领域的基础。
（2） 在特定条件下， Lasso可以recover准确的非零权重。
（3） Lasso使用L1正则项对回归模型的参数进行惩罚： $min||Xw-y||_2^2+\alpha||w||_1$
 (4) Lasso的参数 $alpha$可以通过交叉验证来设置：LassoCV和LassoLarsCV。 LassoLarsCV是基于最小角回归算法的。
 (5) 对于高维的数据，LassoCV是更合适的选择。而LassoLarsCV的优点是可以explore更相关的$alpha$值，并且当样本数量很少时比LassoCV更快。
'''
reg=linear_model.Lasso(alpha=0.1)
reg.fit([[0,0],[1,1]],[0,1])
pre=reg.predict([[1,1]])
print(pre)

'''
1.1.4 多任务Lasso: MultiTaskLasso
(1) MultiTaskLasso是可以为多个回归问题估计参数的线性模型，y为一个2D的数组（n_samples,n_tasks）
(2) 每个任务都使用相同的特征
(3)使用混合的l1l2 prior作为正则项， $l1l2=||A||_{21}=\sum_i\sqrt\sum_j a_{ij}^2$
'''


'''
1.1.5 Elastic Net
(1) ElasticNet是一个线性模型，使用L1和L2作为正则项， 这使得它同时具备Lasso的稀疏性和Ridge的特点。
(2) 当多个特征之间有相互关联时，Elastic-net很有用， Lasso可以随机选取其中的一个特征，而elastic-net可以选取两个(pick both).
(3) 在Lasso和Ridge之间进行权衡可以使 Elastic-Net 继承Ridge的稳定性。
(4) 目标函数是： $min||Xw-y||_2^2+ \alpha \rho||w||_1 + \frac{\alpha(1-\rho)}{2}||w||_2^2$
(5) 可以使用ElasticNetCV通过交叉验证设置参数$\alpha$和$\rho$
'''

'''
1.1.6 Multi-task Elastic Net: 多任务的ElasticNet
(1) MultiTaskElasticNet是为多个回归任务估计稀疏系数的elastic-model
(2) 使用l1l2 prior和l2 prior作为正则项。

注意： l1l2与L1和L2的区别。
'''

'''
1.1.7 Least Angle Regression: 最小角回归
(1) Least Angle Regression(LARS)是处理高维数据的一种回归算法。
(2) 优点:
    * 当p>>n时效率高，（比如，当数据的维度远远大于数据的个数时）
    * 时间复杂度与普通最小二乘法一样
    * 产生了一个分段的线性解决方案，在交叉验证和tune model(调整模型)时很有用。
    * If two variables are almost equally correlated with the response, then their coefficients should increase at approximately the same rate. The algorithm thus behaves as intuition would expect, and also is more stable.
    * 很容易修改来实现其他的模型，比如LASSO
（3）缺点：
    * 由于LARS是基于迭代的重新拟合残差的， 它对噪声的影响比较敏感。
'''


'''
1.1.8 是一种使用LARS算法实现的LASSO, 与基于坐标下降(coordinate_descent)的实现不同， 它产生分段的线性的solution.
'''
reg=linear_model.LassoLars(alpha=0.1)
reg.fit([[0,0],[1,1]],[0,1])
print(reg.coef_)






















