import numpy as np
import matplotlib.pyplot as plt

def initialisation(X):
  W=np.random.randn(X.shape[1],1)
  b=np.random.randn(1)
  return(W,b)


def model(X,W,b):
  Z=X.dot(W)+b # .dot() pour le produit matriciel, library numpy
  A=1/(1+np.exp(-Z))
  return(A)


def log_loss(A,y):
  return -1/len(y)*np.sum(y*np.log(A) + (1-y)*np.log(1-A) )


def gradients(A,X,y):
  dW=1/len(y)*np.transpose(X).dot(A-y) # on peut aussi écrire np.dot(X.T,A-y)
  db=1/len(y)*np.sum(A-y)
  return(dW,db)


def update(dW,db,W,b,learning_rate):
  W=W-learning_rate*dW
  b=b-learning_rate*db
  return(W,b)


def predict(X,W,b):
  A=model(X,W,b)
  #print(A) # la probabilité d'appartenance à la classe 1 (plante toxique)
  return A>=0.5 # retourne 1 si A >= 1/2


from sklearn.metrics import accuracy_score
def artificial_neuron(X,y,learning_rate=0.1,n_iter=100):
  #initialisation
  W,b=initialisation(X)
  Loss=[] #pour stocker les valeurs de coût à chaque itération, et ainsi suivre l'exécution de l'algo
  for i in range(n_iter):
    A=model(X,W,b)
    Loss.append(log_loss(A,y))
    dW,db=gradients(A,X,y)
    W,b=update(dW,db,W,b,learning_rate)
  
  y_pred=predict(X,W,b) # prédiction pour toutes les données du modèle
  print("Cette nouvelle plante a %d" %(accuracy_score(y,y_pred)*100),"% de chance d'être dans la classe 1, càd d'être toxique." )

  plt.plot(Loss)
  plt.show()
  return(W,b)
