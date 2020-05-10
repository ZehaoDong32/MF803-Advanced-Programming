import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# problem 1
tickers = "XLB XLE XLF XLI XLK XLP XLU XLV XLY"
start = "2010-01-01"
end = "2019-11-20"
data = yf.download(tickers, start, end)['Close']

#(a)(b)
ret = data.pct_change()
cov = ret.cov().to_numpy()

#(c)
eigenvalue = np.linalg.eig(cov)[0]    
df_eigen = pd.DataFrame(eigenvalue, 
                             index=['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
                            columns=['eigenvalue'])

sorted_eigen = df_eigen.sort_values(by=['eigenvalue'], ascending=False)

plt.figure(figsize=(6, 4), dpi=120)
plt.plot(sorted_eigen)
plt.title('Eigenvalues for Each ETFs')
plt.xlabel('ETFs')
plt.ylabel('Eigenvalues')
plt.show()

#(d)
X = np.random.normal(0, 1, 9**2).reshape(9, 9)
X = np.triu(X)
X += X.T - np.diag(X.diagonal())

#(e)
eigenvalue2 = np.linalg.eig(X)[0]
sorted_eigen2 = pd.DataFrame(np.sort(eigenvalue2[::-1])[::-1], columns=['eigenvalue'])

plt.figure(figsize=(6, 4), dpi=120)
plt.plot(sorted_eigen2)
plt.title('Simulated Eigenvalues')
plt.ylabel('Eigenvalues')
plt.show()
    
# problem 2
#(a)
a_ret = np.array(ret.mean() * 252)

#(b)
def weights(a, cov, ret):
    w = 1 / (2 * a) * np.dot(np.linalg.inv(cov), ret)
    w = pd.DataFrame(w, index=['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
                     columns=['weights'])
    return w

w0 = weights(1, cov, a_ret)

plt.figure(figsize=(6, 4), dpi=120)
plt.plot(w0)
plt.title('Weights for sigma=0')
plt.xlabel('ETFs')
plt.ylabel('Weights')
plt.show()

#(c)
def expected_r(sigma, ret):
    r = ret + sigma * np.random.normal(0, sigma, 9)
    return r

ret1 = expected_r(0.005, a_ret)
w1 = weights(1, cov, ret1)
ret2 = expected_r(0.01, a_ret)
w2 = weights(1, cov, ret2)
ret3 = expected_r(0.05, a_ret)
w3 = weights(1, cov, ret3)
ret4 = expected_r(0.1, a_ret)
w4 = weights(1, cov, ret4)

plt.figure(figsize=(6, 4), dpi=120)
plt.title('Weights for different sigmas')
plt.xlabel('ETFs')
plt.ylabel('Weights')
plt.plot(w0, color='black', label='sigma=0')
plt.plot(w1, color='skyblue', label='sigma=0.005')
plt.plot(w2, color='b', label='sigma=0.01')
plt.plot(w3, color='r', label='sigma=0.05')
plt.plot(w4, color='green', label='sigma=0.1')
plt.legend()
plt.show()

#(d)
def regularized_cov(delta, diag, cov):
    result = delta * diag + (1 - delta) * cov
    return result

var = np.array(ret.var() * 252)
diag = np.diag(var)

#(e)
reg_cov = regularized_cov(1, diag, cov)
eigen = pd.DataFrame(np.linalg.eig(reg_cov)[0], 
                     index=['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
                     columns=['eigenvalue'])

#(f)
cov_list = []
eigen_df = pd.DataFrame(index=['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'])
for c in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    a = regularized_cov(c, diag, cov)
    cov_list.append(a)
    b = pd.DataFrame(np.linalg.eig(a)[0], 
                         index=['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
                         columns=['eigenvalue' + str(c)])
    eigen_df = eigen_df.join(b)
    
#(g)
reg1_w1 = weights(1, cov_list[0], ret1)
reg1_w2 = weights(1, cov_list[0], ret2)
reg1_w3 = weights(1, cov_list[0], ret3)
reg1_w4 = weights(1, cov_list[0], ret4)
plt.figure(figsize=(6, 4), dpi=120)
plt.title('Weights for different sigmas when delta=0')
plt.xlabel('ETFs')
plt.ylabel('Weights')
plt.plot(reg1_w1, color='skyblue', label='sigma=0.005')
plt.plot(reg1_w2, color='b', label='sigma=0.01')
plt.plot(reg1_w3, color='r', label='sigma=0.05')
plt.plot(reg1_w4, color='green', label='sigma=0.1')
plt.legend()
plt.show()

reg2_w1 = weights(1, cov_list[2], ret1)
reg2_w2 = weights(1, cov_list[2], ret2)
reg2_w3 = weights(1, cov_list[2], ret3)
reg2_w4 = weights(1, cov_list[2], ret4)
plt.figure(figsize=(6, 4), dpi=120)
plt.title('Weights for different sigmas when delta=0.4')
plt.xlabel('ETFs')
plt.ylabel('Weights')
plt.plot(reg2_w1, color='skyblue', label='sigma=0.005')
plt.plot(reg2_w2, color='b', label='sigma=0.01')
plt.plot(reg2_w3, color='r', label='sigma=0.05')
plt.plot(reg2_w4, color='green', label='sigma=0.1')
plt.legend()
plt.show()

reg3_w1 = weights(1, cov_list[4], ret1)
reg3_w2 = weights(1, cov_list[4], ret2)
reg3_w3 = weights(1, cov_list[4], ret3)
reg3_w4 = weights(1, cov_list[4], ret4)
plt.figure(figsize=(6, 4), dpi=120)
plt.title('Weights for different sigmas when delta=0.8')
plt.xlabel('ETFs')
plt.ylabel('Weights')
plt.plot(reg3_w1, color='skyblue', label='sigma=0.005')
plt.plot(reg3_w2, color='b', label='sigma=0.01')
plt.plot(reg3_w3, color='r', label='sigma=0.05')
plt.plot(reg3_w4, color='green', label='sigma=0.1')
plt.legend()
plt.show()

reg4_w1 = weights(1, cov_list[5], ret1)
reg4_w2 = weights(1, cov_list[5], ret2)
reg4_w3 = weights(1, cov_list[5], ret3)
reg4_w4 = weights(1, cov_list[5], ret4)
plt.figure(figsize=(6, 4), dpi=120)
plt.title('Weights for different sigmas when delta=1')
plt.xlabel('ETFs')
plt.ylabel('Weights')
plt.plot(reg4_w1, color='skyblue', label='sigma=0.005')
plt.plot(reg4_w2, color='b', label='sigma=0.01')
plt.plot(reg4_w3, color='r', label='sigma=0.05')
plt.plot(reg4_w4, color='green', label='sigma=0.1')
plt.legend()
plt.show()