import regression, datapreprocess
import numpy as np
import os
import math

if __name__ == '__main__':
    rdc = datapreprocess.RegressionDataContainer(0.7)
    img_mean = np.load('data/img_mean.npz')['arr_0']
    # print(img_mean)
    gdpregression = regression.EconomicRegression(rdc.getimgsize(), 3, img_mean, 'net_structure.json', 'tweights.npz', 'gdpregression_weights.npz')
    gdp_y_predict, gdp_loss, gdp_r2 = gdpregression.test_loss(rdc.data, rdc.y[:, 0:1])
    faregression = regression.EconomicRegression(rdc.getimgsize(), 3, img_mean, 'net_structure.json', 'tweights.npz', 'faregression_weights.npz')
    fa_y_predict, fa_loss, fa_r2 = faregression.test_loss(rdc.data, rdc.y[:, 1:2])
    crvregression = regression.EconomicRegression(rdc.getimgsize(), 3, img_mean, 'net_structure.json', 'tweights.npz', 'crvregression_weights.npz')
    crv_y_predict, crv_loss, crv_r2 = crvregression.test_loss(rdc.data, rdc.y[:, 2:])
    with open('regression_result.txt', 'w') as f:
        for i in range(len(rdc.sample_path)):
            f.writelines(rdc.sample_path[i] + ':' + '\n')
            economic_str = 'GDP:%f, FA:%f, CRV:%f' %(rdc.y[i,0], rdc.y[i,1], rdc.y[i,2])
            pred_economic_str = 'GDP:%f, FA:%f, CRV:%f' %(gdp_y_predict[i,0], fa_y_predict[i,0], crv_y_predict[i,0])
            f.writelines('y_predict: ' + pred_economic_str + '\n')
            f.writelines('y_real: ' + economic_str + '\n')
        
        rmse_str = 'GDP:%f, FA:%f, CRV:%f' %(math.sqrt(gdp_loss), math.sqrt(fa_loss), math.sqrt(crv_loss))
        r2_str = 'GDP:%f, FA:%f, CRV:%f' %(gdp_r2, fa_r2, crv_r2)
        f.writelines('RMSE: ' + rmse_str + '\n')
        f.writelines('R2' + r2_str + '\n')

    

