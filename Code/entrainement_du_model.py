#python3 /Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/entrainement_du_model.py

__auteur__ = "Paul-Noel Digard"

"""--------------------------------------------------

            Partie 2 : création et entrainement de
            notre modèle de reconnaissance de caractere

--------------------------------------------------------"""

# Imports
################################################################################
import pickle
import tensorflow as tf
import numpy as np
import random
################################################################################

# Constants
################################################################################
#on récupere notre liste de couple (feature/label) construite dans la partie I
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/features_and_labels', 'rb') as fp:
    features_labels = pickle.load(fp)
m = int((90/100)*len(features_labels))
features_labels_train = features_labels[:m]
features_labels_test = features_labels[m:]
h, l = 24, 12 #tailles de nos images dans le dossier clear_letters
N = h*l
################################################################################

#création d'un placeholder (ie variable à laquelle on donnera les valeurs des images)
x = tf.placeholder(tf.float32, [None, N])

#initialisation du poids w et du biais b
W = tf.Variable(tf.zeros([N, 35]))      #35 car 9 chiffres (pas le 0) + 26 lettres
b = tf.Variable(tf.zeros([35]))

#création de notre modele (le but va etre de trouver w et b tq y se rapproche le plus du label associé à x)
y = tf.nn.softmax(tf.matmul(x, W) + b)

#il est maintenant temps de determiner w et b grace à la bdd mnist
#tout d'abord on définie un distance pour déterminer à quel point notre prédiction est loin du label
#on crée un nouveau placeholder pour le label
y_ = tf.placeholder(tf.float32, [None, 35])

#puis on définie la distance entre y_(le label) et y (la prédiction)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#on définie maintenant l'étape d'entrainement de notre modèle
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#on entraine enfin réellement notre modèle
#pour cela on ouvre une "session", c'est la procédure dans tensorflow
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#on fait tourner 1000 fois l'algo de descente de gradient
for i in range(1000):
    random.shuffle(features_labels_train)
    batch_xs, batch_ys = np.array(list(zip(*features_labels_train))[0]), np.array(list(zip(*features_labels_train))[1])
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print("Loss {}/1000 : {}".format(i + 1, sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})))

#évaluation du modèle
#évaluer une seule image :
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#prédire une seule image
prediction = tf.argmax(y)

#déterminer l'accuracy :
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Finalement, on observe l'accuracy sur nos données de test pour vérifier notre modèle
features_test, labels_test = np.array(list(zip(*features_labels_test))[0]), np.array(list(zip(*features_labels_test))[1])

print("Accuracy : {}".format(100 * sess.run(accuracy, feed_dict={x: features_test, y_: labels_test})))
# on a un score de 93%

#on sauvegarde les poids (W) et le biais (b) pour les réutiliser dans utilisation_du_model.py
#sans avoir à refaire tourner l'entrainement à chaue fois
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/weigths', 'wb') as fp:
    pickle.dump(sess.run(W), fp)
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/bias', 'wb') as fp:
    pickle.dump(sess.run(b), fp)
