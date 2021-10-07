import matplotlib.pyplot as plt
import numpy as np
import math

loss1 = np.load('results_dyna_500.npy')
loss2 = np.load('results_VTR_500.npy')
plt.plot(loss1, label="Linear Dyna")
standard_error_1 = np.std(loss1)/ math.sqrt(30)
plt.fill_between(range(len(loss1)), loss1-standard_error_1, loss1+ standard_error_1 ,alpha=0.3)
standard_error = np.std(loss2)/ math.sqrt(30)
plt.plot(loss2, color= "red",label= "VTR")
plt.fill_between(range(len(loss2)), loss2-standard_error, loss2+ standard_error, color="red" ,alpha=0.3)
plt.xlabel("Number of Episodes")
plt.ylabel("RMSE(Between analytical and predicted vals)")
plt.legend()
plt.yscale('log')
# loss = np.load('model_loss.npy')
# plt.plot(loss)
plt.show()
