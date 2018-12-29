#python3 /Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/utilisation_du_model.py

__auteur__ = "Paul-Noel Digard"

"""--------------------------------------------------

            Partie 3 : création de la fonction de prédiction

--------------------------------------------------------"""

# Imports
################################################################################
import numpy as np
from PIL import Image
import os
import os.path
import glob
import random
import pickle
################################################################################

# Constants
################################################################################
outputs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#on récupère tout d'abord les poids (W) et le biais (b) du modèle que l'on vient d'entrainer
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/weigths', 'rb') as fp:
    W = np.array(pickle.load(fp))
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/bias', 'rb') as fp:
    b = np.array(pickle.load(fp))
################################################################################

#on construit ensuite toute les fonctions dont on va se servir
def softmax(y):
    n = len(y)
    z = np.zeros(n)
    S = 0
    for yi in y:
        S+=np.exp(yi)
    for (i, yi) in enumerate(y):
        zi = np.exp(yi)/S
        z[i] = zi
    return(z)

def formatage_captcha(img): #prend en entrée un chemin vers une image et renvoie un tableau correspondant en noir et blanc (format 0 ou 1)
    tab_img = np.array(Image.open(img))
    h, l, r = tab_img.shape
    tab_img_noir_blanc = np.zeros((h, l), dtype='float32')
    for i in range(h):
        for j in range(l):
            pixel = int(0.299*tab_img[i][j][0] + 0.587*tab_img[i][j][1] + 0.114*tab_img[i][j][2])
            if pixel>200:
                tab_img_noir_blanc[i][j] = 255
            else:
                tab_img_noir_blanc[i][j] = 0
    return(tab_img_noir_blanc)

def decoupage(captcha): #prend en entrée un captcha sous forme de tableau et renvoie les caracteres du captcha sous forme de tableau
    h, l = captcha.shape
    frontieres = []  #liste contenant les indices des colonnes constituant une frontieres entre deux caracteres
    blanc = True
    noir = False
    for j in range(l):
        if blanc:
            i=0
            while blanc and i<h:
                if captcha[i][j] == 0:
                    frontieres.append(j)
                    blanc = False
                    noir = True
                i=i+1
        else: #cad si noir, on cherche la frontiere de droite
            i = 0
            ok = True   #si ok alors on a que des pixels blancs sur la colonne
            while ok and i<h:
                if captcha[i][j] == 0:
                    ok = False
                i+=1
            if ok:
                frontieres.append(j)
                blanc = True
                noir = False
    #on connait maintenant les indices des contours des caracteres du captcha, on peut procéder au découpage
    split_captcha = []
    for i in range(0, len(frontieres)-1, 2):
        fg, fd = frontieres[i], frontieres[i+1] #il s'agit des indices des colonnes encerclant le premier caractère (ou groupe de caracteres)
        if (fd - fg) < 0.9*h:    #dans ce cas cela signifie qu'il n'y a qu'un seul caractere (et pas deux collés)
            split_captcha.append(np.array([captcha[i][fg-1:fd+1] for i in range(h)]))
        else: #ici on traite le cas où deux caracteres sont collés et n'ont pu etre séparés
            m = int((fd + fg)/2)
            split_captcha.append(np.array([captcha[i][fg-1:m+1] for i in range(h)]))
            split_captcha.append(np.array([captcha[i][m:fd+1] for i in range(h)]))
#on associe maintenant chaque lettre à son label
    if len(split_captcha) == 4:  #sinon cela signifie que le découpage n'a pas bien marché
        letters = []
        for k in range(4):
            img = Image.fromarray(split_captcha[k])
            img = img.resize((12, 24), Image.ANTIALIAS)
            letters.append(np.array(img))
        return(letters)
    else:
        return('Trop compliqué à découper')

def ready_to_test(img):  #passage d'un tableau à une liste et normalisation
    L = []
    h, l = img.shape
    for i in range(h):
        for j in range(l):
            if img[i][j]>150:
                L.append(1)
            else:
                L.append(0)
    return(np.array(L, dtype='float32'))


#on peut enfin coder notre fonction de prédiction
def predictor(captcha):     #en entrée on attend un chemin vers un captcha dont on veut prédire le texte
    captcha_noir_blanc = formatage_captcha(captcha)
    letters = decoupage(captcha_noir_blanc)
    if isinstance(letters, str):
        return('Trop compliqué à décoder')
    else:
        lettres_a_predire = []
        for letter in letters:
            lettres_a_predire.append(ready_to_test(letter))
        prediction = ''
        for x in lettres_a_predire:
            y = softmax(np.dot(x, W) + b)
            prediction+= outputs[y.tolist().index(max(y))]
        return(prediction)


print("Le texte du captcha est {}".format(predictor('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/captchas/2H2Z.png')))
