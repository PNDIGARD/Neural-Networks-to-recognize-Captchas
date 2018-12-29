#python3 /Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/creation_bdd_propre.py

__auteur__ = "Paul-Noël Digard"

"""------------------------------------------

                Partie I : création de la bdd
        (découpage des captchas et association avec leur label)

-----------------------------------------------"""

"""
1ere étape : Associer à chaque captcha de la base son texte (ie son label), puis séparer notre nouvelle base (features/labels)
en un train set et un test set.
"""
import os
import os.path
import glob
import random
import pickle

#tout d'abord on récupère les captchas dans le dossier "captchas" (il y en a 9 956)
captchas_list = glob.glob("/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/captchas/*.png") #liste de chemins vers chaque captchas

#ensuite on attribue le label du captcha à son chemin
captchas_and_labels = []
for captcha in captchas_list:
    filename = os.path.basename(captcha)
    label = os.path.splitext(filename)[0]
    captchas_and_labels.append((captcha, label))

#captchas_and_labels est maintenant une liste de couple chemin vers le captcha/texte du captcha (ie label du captcha)

#On va maintenant découper notre base en un train set et un test set
random.shuffle(captchas_and_labels)
captchas_train = captchas_and_labels[:900]
captchas_test = captchas_and_labels[900:]

with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/captchas_test', 'wb') as fp:
    pickle.dump(captchas_test, fp)

#On travail desormais uniquement sur captchas_train
"""
2eme étape : On va découper tout les captchas de manière à isoler chaque (caractère, label). Ces (caractères, labels) vont
etre utilisé pour entrainer notre fonction de reconnaissance de caractere.
"""
import numpy as np
from PIL import Image

#ouverture et transformation des images png de captcha en tableaux
h, l, r =0, 0, 0
captchas_sous_forme_de_tableaux = []
for (i, (captcha_path,lbl)) in enumerate(captchas_train):
    print("[mise sous forme de tableau] processing captcha {}/{}".format(i + 1, len(captchas_train)))
    captcha = Image.open(captcha_path)  #on ouvre le captcha correspondant au chemin captcha_path
    tab_captcha = np.array(captcha)     #on le convertie en tableau de type RVB 0-255
    h,l,r=tab_captcha.shape             #on récupere les dimensions des captchas (ils sont tous de même taille)
    captchas_sous_forme_de_tableaux.append((tab_captcha, lbl))

#transformation des tableaux RVB au format teintes de gris
captchas_gris = []
for (i, (tab_captcha, lbl)) in enumerate(captchas_sous_forme_de_tableaux):
    print("[niveaux de gris] processing captcha {}/{}".format(i + 1, len(captchas_sous_forme_de_tableaux)))
    tab_captcha_gris = np.zeros((h, l), dtype="uint8")
    for i in range(h):
        for j in range(l):
            tab_captcha_gris[i][j] = int(0.299*tab_captcha[i][j][0] + 0.587*tab_captcha[i][j][1] + 0.114*tab_captcha[i][j][2])
    captchas_gris.append((tab_captcha_gris, lbl))

#transformation du format teintes de gris au format noir blanc
captchas_noirs = []
for (i, (tab_captcha_gris, lbl)) in enumerate(captchas_gris):
    print("[noir et blanc] processing captcha {}/{}".format(i + 1, len(captchas_gris)))
    captcha_noir = np.zeros((h, l), dtype="uint8")
    for i in range(h):
        for j in range(l):
            if tab_captcha_gris[i][j]>200:
                captcha_noir[i][j] = 255
            else:
                captcha_noir[i][j] = 0
    captchas_noirs.append((captcha_noir, lbl))

#découpage des captchas (ie création d'une bdd (lettre/label))
letters = []
for (i, (captcha_noir, lbl)) in enumerate(captchas_noirs):
    print("[découpage] processing captcha {}/{}".format(i + 1, len(captchas_noirs)))
    frontieres = []  #liste contenant les indices des colonnes constituant une frontieres entre deux caracteres
    blanc = True
    noir = False
    for j in range(l):
        if blanc:
            i=0
            while blanc and i<h:
                if captcha_noir[i][j] == 0:
                    frontieres.append(j)
                    blanc = False
                    noir = True
                i=i+1
        else: #cad si noir, on cherche la frontiere de droite
            i = 0
            ok = True   #si ok alors on a que des pixels blancs sur la colonne
            while ok and i<h:
                if captcha_noir[i][j] == 0:
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
            split_captcha.append(np.array([captcha_noir[i][fg-1:fd+1] for i in range(h)]))
        else: #ici on traite le cas où deux caracteres sont collés et n'ont pu etre séparés
            m = int((fd + fg)/2)
            split_captcha.append(np.array([captcha_noir[i][fg-1:m+1] for i in range(h)]))
            split_captcha.append(np.array([captcha_noir[i][m:fd+1] for i in range(h)]))
#on associe maintenant chaque lettre à son label
    if len(split_captcha) == 4:  #sinon cela signifie que le découpage n'a pas bien marché, on utilisera donc pas ce captcha pour entrainer notre modele car il est de mauvaise qualité
        for k in range(4):
            letters.append((split_captcha[k], lbl[k]))

import pickle
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/letters_pckl', 'wb') as fp:
    pickle.dump(letters, fp)

#on enregistre notre bdd d'entrainement pour notre modele de reconnaissance de caractère
for (i, (letter, lbl)) in enumerate(letters):
    print("[création bdd letters] processing letter {}/{}".format(i + 1, len(letters)))
    im = Image.fromarray(letter)
    path = "/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/letters/" + lbl + '(' + str(i) + ')' + ".png"
    im.save(path)

"""
3eme étape : redimensionner nos image pour avoir une bdd utilisable
"""
#il reste à redimensionner ttes les lettres de la bdd letters pour qu'elles aient ttes la meme dimensions
#tout d'abord on détermine la taille moyenne des lettres que l'on vient d'extraire
ls = []  #tableau regroupant les largeurs des lettres
for (letter, lbl) in letters:
    ls.append(letter.shape[1])
mean_l = int(np.mean(ls))
mean_h = h

#on redimensionne alors toute les lettres en dimension (mean_h, mean_l)
list_letters = glob.glob("/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/letters/*.png")
for (i, letter) in enumerate(list_letters):
    print("[redimensionnement] processing letter {}/{}".format(i + 1, len(list_letters)))
    img = Image.open(letter)
    img = img.resize((mean_l, mean_h), Image.ANTIALIAS)
    name = os.path.basename(letter)
    path = "/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/letters_resized/" + name
    img.save(path)

#on réalise qu'apres redimensionnement, certaines lettres ont des contour un peu flou.
#on va donc leur appliquer le même traitement que celui pour passer de teintes de gris à noir et blanc
list_blurred_letters = glob.glob("/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/letters_resized/*.png")
for (i, letter) in enumerate(list_blurred_letters):
    print("[défloutage] processing letter {}/{}".format(i + 1, len(list_blurred_letters)))
    img = Image.open(letter)
    tab_img = np.array(img)
    clear_img = np.zeros((mean_h, mean_l), dtype="uint8")
    for i in range(mean_h):
        for j in range(mean_l):
            if tab_img[i][j]>150:
                clear_img[i][j] = 255
            else:
                clear_img[i][j] = 0
    name = os.path.basename(letter)
    path = "/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/clear_letters/" + name
    Image.fromarray(clear_img).save(path)

"""
4eme étape : mettre les images des lettres et leur label sous forme de liste
"""

h, l = 24, 12 #tailles de nos images dans le dossier clear_letters

outputs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

features_labels = []
list_features = glob.glob("/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/clear_letters/*.png")
for (ii, feature) in enumerate(list_features):
    print("[conversion letters and label to lists] processing letter {}/{}".format(ii + 1, len(list_features)))
    img = Image.open(feature)
    tab = np.array(img).tolist()
    lettre_sous_forme_de_liste = []
    for i in range(h):
        for j in range(l):
            lettre_sous_forme_de_liste.append(tab[i][j]/255)
    lettre_sous_forme_de_liste = np.array(lettre_sous_forme_de_liste, dtype='float32')
    label_dict = { _ : 0 for _ in outputs}
    label = list(os.path.basename(feature))[0]
    label_dict[label] = 1
    label_vect = np.array(list(label_dict.values()))
    features_labels.append((lettre_sous_forme_de_liste, label_vect))

#on sauvegarde features_labels sur le disque pour pouvoir les utiliser dans la partie II : entrainement du model
import pickle
with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/features_and_labels', 'wb') as fp:
    pickle.dump(features_labels, fp)

#On a desormais une bdd (features_labels) propre avec un label pour chaque caractere sous forme de liste. Pret à l'emploi pour
#entrainer notre modèle
