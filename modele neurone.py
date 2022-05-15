from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utilities import *

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
  print(accuracy_score(y,y_pred))

  plt.plot(Loss)
  plt.show()
  return(W,b)
