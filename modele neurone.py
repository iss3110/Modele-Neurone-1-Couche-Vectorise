from sklearn.datasets import make_blobs
from utilities import *

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0],1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

W,b=artificial_neuron(X,y)
new_plant = np.array([2,1])

predict(new_plant,W,b)
# La frontière de décision corréspond à A=1/2 ou bien Z=0,
# cad w1.x1 + w2.x2 + b = 0
# ou x2 = -(w1.x1 + b)/w2
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
x1=np.linspace(-1,4,100) # un vecteur entre -1 et 4 avec 100 valeurs
x2=-(W[0]*x1+b)/W[1]
plt.plot(x1,x2,c="blue",lw=3)
plt.scatter(new_plant[0], new_plant[1],c="r")
plt.show() # remarque que cette ligne qui gère l'affichage du plot, donc permet d'afficher différents plot sur le même graphe ou non