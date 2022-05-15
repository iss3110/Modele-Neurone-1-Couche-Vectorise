# Modele-Neurone-1-Couche-Vectorise
## une intro aux réseaux de neurones

Le machine learning est un domaine de l'inteligence artificielle, qui consiste à programmer une machine pour que celle-ci aprenne à réaliser des tâches en étudiant des exemples de ces dernières. Mathématiquement, ces exemples sont représentés par des données que la machine utilise pour développer un modèle, par exemple f(x)=a.x+b, le but en machine learning, est de trouver les paramètres a et b qui donnent le meilleur modèle possible, c-à-d le modèle qui s'ajuste le mieux aux données. Pour cela, on programme dans la machine un algo d'optimisation qui va venir tester différentes valeurs de a et b jusqu'à obtenir la combinaison de a et b qui minimise la distance entre le modèle et les données, c'est le machine learning : développer un modèle en se servant d'un algo d'optimisation pour minimiser les erreurs entre le modèle et nos données.

![image](https://user-images.githubusercontent.com/78855647/168451625-a76007c2-b0ac-4f86-88b9-052f2ee85149.png)

Les modèles sont infinis, on en peut citer : les modèles linéaires, Arbres de décision, SVM (Support Vector Machine) ... et pour chaque modèle, on doit trouver un algo d'optimisation adapté :

![image](https://user-images.githubusercontent.com/78855647/168451644-d91af889-6554-422a-afa0-b4bb4ac75371.png)

### Deep learning :

![image](https://user-images.githubusercontent.com/78855647/168451663-ff84e3ec-3d76-4a0e-8b71-14d373411cca.png)

Domaine du machine learning, dans lequel au lieu de développer un des modèles que l'on vient de citer par exemple, on développe à la place des Réseaux de Neurones Artificiels, le principe reste exactement le même, c-à-d que l'on fournit à la machine des données, et à l'aide d'un algo d'optimisation, on ajuste le modèle à ces données, mais le modèle n'est pas une fonction explicite comme dans les cas d'avant, ici c'est un réseau de fonctions connectées les unes aux autres : un Réseau de Neurones.

<img width="672" alt="image" src="https://user-images.githubusercontent.com/78855647/168451697-f8c57832-5ddd-4085-bb1d-7f703d0a5f3d.png">

*Remarque :* Plus le réseau est **profond** c-à-d plus il contient des fonctions à l'intérieur, plus la machine est capable de réaliser des tâches complexes : reconnaitre des objets, identifier une personne sur une photo, conduire une voiture etc ... d'où l'on parle **d'apprentissage profond** ou **Deep Learning** lorsqu'on développe des réseaux de neurones artificiels.

![image](https://user-images.githubusercontent.com/78855647/168451735-c2e881a1-d6ca-4688-bb44-a2c2f270d4cf.png)

### Histoire du deep learning :

Pour bien comprendre le fonctionnement des réseaux de neurones artificiels (RNA), on revient un peu sur l'origine de leur histoire, comment ils ont été inventé, et qu'elles furent leurs évolutions à travers le temps, pour arriver à la technologie que nous connaissons ajd'hui. Les 1ers RNA ont été inventé en 1943 par 2 mathématiciens et neuro scientifiques : Warren McCulloch et Walter Pitts, dans leur article intitulé "A logical calculus of the ideas immanent in nervous activity ils expliquent comment ils ont crées des neurones artificiels en s'inspirant des neurones biologiques.

<img width="551" alt="image" src="https://user-images.githubusercontent.com/78855647/168451776-8a99543f-2b0c-4e3c-862f-321418f98f2a.png">

En biologie, les neurones sont des cellules excitables connectées les unes aux autres et ayant pour rôle de transmettre des infos dans notre système nerveux. Chaque neurone est composé de plusieurs dendrites (portes d'entrée), d'un corps cellulaire, et d'un axone. Dendrites : c'est à cet endroit au niveau de la synapse que le neurone reçoit des signaux lui provenant des neurones qui le précèdent, ces signaux peuvent être de type excitateur (+1) ou inhibiteur (-1). Lorsque la somme de ces signaux dépasse un certain seuil, le neurone s'active, et produit alors un signal éléctrique, ce signal circule le long de l'axone en directions des terminaisons pour être envoyé à son tour vers d'autres neurones de notre système nerveux, neurones qui fonctionneront exactement de la même manière.

Ce que Warren McCuloch et Walter Pitts ont essayé de faire, c'est de modéliser ce fonctionnement en considérant qu'un neurone pouvait être représenté par une ***fonction de transfert*** qui prend en entrée des signaux (x)ᵢ et qui retourne une sortie y. A l'intéreur de cette fonction, on trouve deux grandes étapes : 


1.   Agrégation : 𝐟 = ∑ wᵢ.xᵢ
2.   Activation : y = 1 si f >= 0, et y = 0 sinon.

<img width="586" alt="image" src="https://user-images.githubusercontent.com/78855647/168451826-e7e77cce-9418-4fbe-ab0f-d68e471418e8.png">

Ce modèle historique d'un neurone artificiel est conçu pour traiter des entrées logiques (0 ou 1) d'où son appelation **Treshold Logic Unit**
Dans leur article, ils ont pu démontrer qu'avec ce modèle, on peut reproduire certaines fonctions logiques tel que la porte And, et la porte Or, ils ont égaement démontrer qu'en connectant plusieurs de ces fonctions les unes aux autres, un peu comme la manière des neurones de notre cerveau, alors il serait possible de résoudre n'importe quel problème de logique booléenne. Par contre ce modèle de deep leaning, ne dispose pas d'un algo d'apprentissage, et il faut donc trouver nous même les valeurs des paramètres wᵢ si l'on désire s'en servir pour des applications du monde réel.
15 ans plutard, en 1957, Franck Rosenblatt (psycholoque américain) invente le ***Perceptron*** qui permet d'améliorer ces modéles de deep learning en proposant un 1er algorithme d'apprentissage de l'histoire du deep learning.

<img width="614" alt="image" src="https://user-images.githubusercontent.com/78855647/168451880-f7be4f1d-68f0-4efe-b6f2-e7871aa88207.png">

Suite à ça, il y eu un engouement démesuré pour l'IA, on pensait qu'on pouvait grâce aux perceptrons développer des macines capables de lire, de parler.. et même d'avoir une conscience, mais tous cet engouement s'effondra quelques années plus tard lorsqu'on s'est rendu compte que c'est faux, en partie prcq le perceptron est un modèle linéaire. On connut alors le 1er hiver de l'IA, où l'on a quasiment arrêter de financer les recherches en IA.

### Perceptron multi-couche (1986) :

Après cette 1ere période de froid, Geoffrey Hinton, l'un des pères du deep learning développa en 1986 un Perceptron Multichouche, le 1er véritable RNA

### Perceptron :

Le modèle de perceptron ressemble de très près au 1er modèle, et dispose d'un algo d'apprentissage lui permettant de trouver les valeurs des wᵢ afin de trouver les valeurs de sortie y qui nous conviennent.

<img width="638" alt="image" src="https://user-images.githubusercontent.com/78855647/168451934-6250f5cf-7214-4c85-ae42-4323fca5c3de.png">

Pour développer cet algo d'apprentissage, Rosenblatt s'est inspiré de la théorie de Hebb :

<img width="628" alt="image" src="https://user-images.githubusercontent.com/78855647/168451945-9f4e27a9-7267-4149-b83e-f62f46438e4b.png">

A partir de cette idée de renforcement, Rosenblatt a développé un algo d'apprentissage qui consiste à entrainer un neurone artificiel sur des *données de référence* (X,y) pour que celui ci renforce ces paramètres W à chaque fois qu'une entrée X est activée en même temps que la sortie y présente dans ces données, et pour ça il a imaginé la formule suivante dans laquel W sont mis à jour en calculant la différence entre la sortie de référence et la sortie produite par le neurone et en multipliant cette différence par la valeur de chaque entrée X ainsi que par un pas d'apprentissage positif \:

<img width="596" alt="image" src="https://user-images.githubusercontent.com/78855647/168451954-2ef818f2-b9de-46f4-bb36-31ef7a9d984b.png">

De cette manière, si notre neurone produit une sortie différente de celle qu'il est censé produire, par exemple il nous sort y=0 alors qu'on veut y=1 alors notre formule donnera W=W+αX, donc pour les entrées x qui valent 1, le coéfficient W se verra augementé d'un petit pas α, il sera **renforcé**, ce qui provoquera une augementation de la fonction f=w1.x1+w2.x2, et qui rapprochera donc notre neurone de son seuil d'activation.
Aussi longtemps que l'on sera en dessous de ce seuil, cad aussi longtemps que le neurone produira une mauvaise sortie, alors le coéff w continuera d'augementer grâce à notre formule jusqu'au moment où y_true vaudra y et à ce moment là notre formule donnera W=W+0, ce qui veut dire que nos paramètres arrêterons d'évoluer. C'est ainsi que Rosenblatt a développé le 1er algo d'apprentissage de l'histoire du deep learning.

<img width="525" alt="image" src="https://user-images.githubusercontent.com/78855647/168451963-e90cae9d-604e-441c-a618-bef09c5d3344.png">

### Exemple de perceptron :

<img width="532" alt="image" src="https://user-images.githubusercontent.com/78855647/168451972-5503ceb0-7bc3-4e34-998b-7dbf0ba4e794.png">

Pour déterminer cette droite de décision, à l'aide d'un modèle linéaire, pertinent dans ce cas, on va fournir les valeurs x1 et x2 à un neurone, et en multipliant chaque entrée du neurone par un poids W (w1, w2), dans ce neurone on va également faire ppasser un coéf complémentaire qu'on appelle le biais, ce qui nous donne \:

<img width="537" alt="image" src="https://user-images.githubusercontent.com/78855647/168451986-b48e3d3a-c00b-4b5e-875e-9734e2021594.png">

Ce qui nous donne l'équation de la frontière de décision z(x1,x2)=0, qui sépare bien les deux groupes de données.

<img width="452" alt="image" src="https://user-images.githubusercontent.com/78855647/168452002-4f621867-547b-439f-8c70-6eba02a89f18.png">

Pour prédire dans quelle classe appartient une future nouvelle plante, il va falloir régler les paramètres W et b pour séparer au mieux les deux classes, et décider de l'appartenance de cette nouvelle plante selon le signe de z \:

<img width="552" alt="image" src="https://user-images.githubusercontent.com/78855647/168452020-38ff40f2-bba7-40b6-b0b6-02c10a1bb60b.png">

Pour améliorer ce modèle, ca serait d'accompagner chaque prédiction d'une probabilité d'appartenance, plus une plante sera éloignée de la frontière de décision, plus il sera évident qu'elle appartienne bien à sa classe, pour ça on pourrait utiliser une *fonction d'activation*, nous retournant une sortie qui s'appproche de 0 ou 1 au fur et au mesure que l'on s'éloigne de la FD (z=0). Cette fonction est appelé *fonction logistique* ou *sigmoîde*.

### Fonction Sigmîde (Logistique) :

Cette fonction permet de convertir la sortie z en une probabilité a(z) \: la proba de l'appartenance à la classe 1,

<img width="527" alt="image" src="https://user-images.githubusercontent.com/78855647/168452042-4f3b1b46-f213-4c58-8e65-77a028a6176c.png">

Pour z=-2.1, a(z)=0.1, cette plante à 10% de proba d'être toxique (classe 1).
Pour z=1.4, a(z)=0.8, 80% d'être toxique.

### Loi de Bernoulli :

Y suit une loi de Bernoulli : 

<img width="307" alt="image" src="https://user-images.githubusercontent.com/78855647/168452071-fcdba182-028c-4c4c-8917-83095693b1d0.png">

Pour résumer, ce qu'on trouve à l'intérieur d'un neuronne c'est une fonction linéaire z = w1.x1+w2.x2+b, suivi d'une fonction d'activation, la plus simple étant la fonction sigmoîde qui nus retourne une proba suivant une loi de Bernoulli.

<img width="541" alt="image" src="https://user-images.githubusercontent.com/78855647/168452550-35915c9d-dd21-4809-afcb-d8f9e2ad52b5.png">

Maintenant, notre but est de régler les paramètres W et b de façon à obtenir le meilleur modèle possible, cad le modèle qui fait les plus petites erreurs entre les sorties a(z) et les vraies données Y. On va définir une *fonction coût* qui va permettre de mesurer ces erreurs.

### Fonction coût (Vraisemblance dans ce cas) :

Une fonction coût (loss function) permet de quantifier les erreurs effectuées par un modèle. Dans notre cas elle permettra de mesurer les distances suivantes \:

<img width="596" alt="image" src="https://user-images.githubusercontent.com/78855647/168452586-53d33902-bbdf-4482-9c47-4397ae81085f.png">

#### Vraisemblance :

Une façon d'évaluer la performance d'un modèle, c'est de calculer sa vraisemblance, en statistique, elle indique la plausiblité du modèle vis à vis de vraies données, par analogie, une histoire est vraisemblable lorsqu'elle est en accord avec des faits réels qui se sont vraiment déroulés.

<img width="521" alt="image" src="https://user-images.githubusercontent.com/78855647/168452602-80ad8122-58d8-4a8c-8539-cb52897bce2d.png">

Ainsi on calcule la vraisemblance de notre modèle, L = ℿ P(Y=yi), pour i= 1 à m, en utilisant la loi de Bernoulli, cela nous donne \:

<img width="557" alt="image" src="https://user-images.githubusercontent.com/78855647/168452618-9fc7df2c-dd0a-4a32-947b-6688dc14bdd4.png">

Si ce résultat est proche de 1 (100%), ca signifiera que notre modèle est vraisemblable à 100%, cad il colle parfaitement aux données que l'on considère vraies.

<img width="541" alt="image" src="https://user-images.githubusercontent.com/78855647/168452629-8ed3bc18-7462-4eb3-b63e-bd6a4e176741.png">

Si L est proche de 0, ça voudrait dire que notre modèle est fortement invraisemblable, si ce modèle existe, cela signifierait que toutes les données dont nous disposons seraient en réalité fausses.

<img width="531" alt="image" src="https://user-images.githubusercontent.com/78855647/168452654-748d6f2a-eb04-4d63-b003-04fa0b97c054.png">

On doit donc trouver une astuce pour calculer cette vraisemblance sans pour autant converger vers 0. C'est pour ça qu'on utilise la fonction log vraisemblance, en effet la log passe de produit à une somme \:

<img width="566" alt="image" src="https://user-images.githubusercontent.com/78855647/168452671-0166ff4e-fe99-4bf3-a61f-22e547284ba1.png">

Le passage en log vraisemblance ne fausse absolument pas les calculs afin de trouver les valeurs de W et b qui maximisent la log vraisamblence, car en effet la fonction log et monotone croissante \:

<img width="384" alt="image" src="https://user-images.githubusercontent.com/78855647/168452682-68dfa19d-4ca7-4469-86c3-649600dd7f43.png">
<img width="580" alt="image" src="https://user-images.githubusercontent.com/78855647/168452688-5232bcfa-1952-4292-ab0c-13361113d3cf.png">

Analytiquement on a :

<img width="524" alt="image" src="https://user-images.githubusercontent.com/78855647/168452697-7ffb4107-0f69-469c-9672-18ee7877db6a.png">

D'où l'explication de la fonction de coût, le facteur 1/m est là pour normaliser le résultat, et le signe (-) car en optimisation les algorithmes connus cherchent à minimiser une fonction, c'est la même chose car maximiser f(x) revient à minimiser -f(x).

![image](https://user-images.githubusercontent.com/78855647/168452707-6591f211-aecb-44b1-aeb8-3f9386921fcf.png)

Pour ce problème d'optimisation, on va utiliser l'algorithme de *la déscente de gradient.*










