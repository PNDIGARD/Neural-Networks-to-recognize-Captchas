#python3 /Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/model_accuracy.py

__auteur__ = "Paul-Noel Digard"

"""--------------------------------------------------

            Partie 4 : calcul de l'efficacité du modele

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
from utilisation_du_model import predictor
################################################################################

def accuracy(liste_path_to_captchas_labeled):
    n_correct = 0
    N = len(liste_path_to_captchas_labeled)
    for (i, (captcha, lbl)) in enumerate(liste_path_to_captchas_labeled):
        print(predictor(captcha), lbl)
        if predictor(captcha) == lbl:  #not isinstance(predictor(captcha), str) and
            n_correct+=1
            print("Captcha {}/{} correctly predicted".format(i + 1, N))
        else:
            print("Captcha {}/{} Fail !".format(i + 1, N))
    return(n_correct/N)

with open('/Users/Paul-Noel/Desktop/ENSAE/Info/Projet_python/captchas_test', 'rb') as fp:
    captchas = pickle.load(fp)

print(accuracy(captchas))   #on a un score de .74 logique car notre algo de reconnaissance des lettres a un score
                            # de .93 et .74 = .93^4 (4 lettres par captcha)
