import pickle as pk
import PIL
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import convolve2d


allfeat={}
allfeat["40X"]=["LBP331full","LBPriv1full","cLBPfull","LPQfull"]
allfeat["100X"]=["LBP33full10","LBPRIVfull10","cLBPfull10","LPQfull10"]
allfeat["200X"]=["LBP33full20","LBPRIV20full","CLBPfull20","LPQfull20"]
allfeat["400X"]=["LBP33full40","LBPrivfull40","CLBPfull40","LPQfull40"]
feat=['lbp','lbpriv','clbp','lpq']
par={}
par["40X"]=[['poly',2,0.1],['poly',10,2],['poly',1,0.09],['poly',5,0.5]]
par["100X"]=[['poly',10,0.1],['poly',1,7],['poly',0.5,0.1],['poly',5,0.1]]
par["200X"]=[['poly',4,0.1],['poly',1,7],['poly',1,0.1],['poly',0.005,0.9]]
par["400X"]=[['poly',1,0.1],['poly',0.5,3],['poly',0.5,0.1],['poly',0.009,0.9]]

def trainfeat(magn):
    trf=[]
    for i in allfeat[magn]:
        tf=i
        f=open(tf,'rb')
        trfi=pk.load(f)
        f.close()
        trf.append(trfi)
    print("training done")
    return trf
    

l1=["40X","100X","200X","400X"]
Dclf=dict()
for j in l1:
    da=trainfeat(j)
    Dclf[j]=[]
    print(j,":")
    for i in range(4):        
        (X,Y) =(da[i][0],da[i][1])
        scaler = MinMaxScaler(feature_range=(0, 1))
        x1 = scaler.fit_transform(X)
        k=par[j][i][0]
        C=par[j][i][1]
        g=par[j][i][2]
        clf=svm.SVC(kernel=k,gamma=g, C=C)
        print(k,"gamma=",g,"c=",C,"not colored",feat[i])
        clf.fit(x1,Y)
        Dclf[j].append(clf)
        
        
f=open('DPclf','wb')
pk.dump(Dclf,f)
f.close()
