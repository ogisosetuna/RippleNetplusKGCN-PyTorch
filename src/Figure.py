#导入必要的模块
import numpy as np
import matplotlib.pyplot as plt


#录入身高与体重数据
#ACC = ['170','179','159','160','180','164','168','174','160','183']
AUC = [0.6261,0.6358,0.6385,0.6306,0.6301]
Dimension = ['2','4','8','16','32']

plt.plot(Dimension, AUC)
plt.xlabel('Dimension')
plt.ylabel('ACC')
plt.title('ACC of different dimension on book dataset')
#plt.title  设置图像标题

#设置纵坐标刻度
plt.yticks([0.625, 0.630, 0.635,0.640])


#plt.fill_between(Dimension, AUC, 10, color = 'green')


plt.show()