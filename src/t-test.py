#导入第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.weightstats as st
from scipy import stats

#读取文件
xls = pd.ExcelFile('t-test.xlsx')
data = xls.parse('Sheet1', dtype='object')

#字符串转换为数值
data['ripple'] = data['ripple'].astype('float')
data['rippleGCN'] = data['rippleGCN'].astype('float')

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10, 5)
sns.kdeplot(data=data['ripple'], legend=False, ax=axes[0])
sns.kdeplot(data=data['rippleGCN'], legend=False, ax=axes[1])

axes[0].set(title='ripple分布')
axes[1].set(title='rippleGCN分布')
plt.subplots_adjust(wspace=0.2)



data.describe()
alpha = 0.1
t, p_two, df = st.ttest_ind(data['ripple'], data['rippleGCN'])


print('alpha = ',alpha)
print('t=' + str(t))
print('P value=' + str(p_two))
print('degree of freedom =' + str(df))


if(p_two < alpha):
    print('P<α，significant。')
else:
    print('P>α，not significant。')