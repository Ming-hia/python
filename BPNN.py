
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing


def getdata():
    data=pd.read_csv("train.csv")
    Y=data.Survived
    Y=np.matrix(Y).T

    X=data
    X.drop("PassengerId",axis=1,inplace=True)
    X.drop("Survived",axis=1,inplace=True)
    X.drop("Name",axis=1,inplace=True)

    X["Sex_female"]=map(int,list(X["Sex"]=="female"))
    X["Sex_male"]=map(int,list(X["Sex"]=="male"))
    X.drop("Sex",axis=1,inplace=True)

    X.drop("Ticket",axis=1,inplace=True)
    X.drop("Cabin",axis=1,inplace=True)
    X.drop("Embarked",axis=1,inplace=True)
    X.Age.fillna(30,inplace=True)

    X_scale=preprocessing.scale(X)
    X_scale=np.matrix(X_scale)

    return X_scale,Y

def sigmoid(x,a=1):
    try:
        e=np.exp(-1*a*x)
    except OverflowError:
        return 0.0
    y=1.0/(1 + e)
    return y

def costfunc(y,a):
    y_l=y.tolist()
    y_l=map(lambda c:c[0],y_l)
    a_l=a.tolist()
    a_l=map(lambda c:c[0],a_l)
    cost=-sum(map(lambda c,d:c*np.log(d)+(1-c)*np.log(1-d),y_l,a_l))/len(y_l)
    return cost

class BPNN():
    def __init__(self,X,Y,hidden=[3,2]):
        self.X=X
        self.Y=Y
        layer_num=len(hidden)
        self.col_num=self.X.shape[1]
        self.row_num=self.X.shape[0]
        self.wlist=[]
        for idx,layer in enumerate(hidden):
            if idx==0:
                self.wlist.append(np.matrix(np.random.random(size=(self.col_num,layer))))
                layer_e=layer
                continue
            self.wlist.append(np.matrix(np.random.random(size=(layer_e+1,layer))))
            layer_e=layer
        self.wlist.append(np.matrix(np.random.random(size=(layer_e+1,1))))

    def forward(self):
        self.zlist=[]
        self.alist=[]
        for idx,w in enumerate(self.wlist):
            if idx==0:
                z=self.X*w
                a=sigmoid(z)
                a=np.hstack((np.matrix(np.ones((self.row_num,1))),a))
                self.zlist.append(z)
                self.alist.append(a)
                continue
            z=self.alist[-1]*w
            a=sigmoid(z)
            if idx != len(self.wlist)-1:
                a=np.hstack((np.matrix(np.ones((self.row_num,1))),a))
            self.zlist.append(z)
            self.alist.append(a)

    def backward(self):
        self.errlist=[]
        self.deltalist=[]
        for i in range(len(self.wlist)):
            if i == 0:
                a=self.alist[-1]
                ab=self.alist[-2]
                err=(a-self.Y)*a.T*(1-a)
            elif i == len(self.wlist)-1:
                w=self.wlist[-i]
                ab=self.X
                err=self.errlist[-1]
                err=err*w.T
                err=err[:,1:]
            else:
                w=self.wlist[-i]
                ab=self.alist[-(i+2)]
                err=self.errlist[-1]
                err=err*w.T
                err=err[:,1:]
            delta=ab.T*err
            self.errlist.append(err)
            self.deltalist.append(delta)

    def updatew(self):
        for i in range(len(self.wlist)):
            delta=self.deltalist[-(i+1)]
            w=self.wlist[i]
            w=w-0.001*delta/self.row_num
            self.wlist[i]=w

    def train(self):
        self.cost_list=[]
        self.forward()
        cost1=costfunc(self.Y,self.alist[-1])
        cost_diff=1
        i=0
        while cost_diff > 0.0001:
            self.backward()
            self.updatew()
            self.forward()
            cost2=costfunc(self.Y,self.alist[-1])
            cost_diff=cost1-cost2
            cost1=cost2
            self.cost_list.append(cost1)
            print i,"|",cost_diff

    def predict(self,X):
        for idx,w in enumerate(self.wlist):
            if idx==0:
                z=X*w
                a=sigmoid(z)
                a=np.hstack((np.matrix(np.ones((self.row_num,1))),a))
                continue
            z=a*w
            a=sigmoid(z)
            if idx != len(self.wlist)-1:
                a=np.hstack((np.matrix(np.ones((self.row_num,1))),a))
        return a



X,Y=getdata()
b=BPNN(X,Y,hidden=[3,2])
b.train()


def judge(y,a):
    y_l=y.tolist()
    y_l=map(lambda c:c[0],y_l)
    a_l=a.tolist()
    a_l=map(lambda c:c[0],a_l)
    a_l=map(lambda c:1 if c> 0.5 else 0,a_l)
    cost=float(sum(map(lambda c,d:1 if c == d else 0,y_l,a_l)))/len(y_l)
    return cost

judge(b.Y,b.alist[-1])


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=0.0001,hidden_layer_sizes=(4,3), random_state=1,activation='logistic' )
x=X.tolist()
y=map(lambda a:a[0],Y.tolist())
clf.fit(x,y)
p=clf.predict(x)
p=p.tolist()
cost=float(sum(map(lambda c,d:1 if c == d else 0,y,p)))/len(y)
print cost




