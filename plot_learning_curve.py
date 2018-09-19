import numpy as np
import pylab
import matplotlib.pyplot as plt
f=open('./fine_val_mean_log.txt',"r")
lines=f.readlines()
score=[]
for line in lines:
    row=line.split(',')
    col=[int(x) for x in row]
    score.append(col[1])
plt.plot(score,label='loss')
plt.xlabel('epoch',fontsize=14)
plt.ylabel('error',fontsize=14)
#plt.title('test')
#plt.legend()
pylab.show()
