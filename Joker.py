# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:26:52 2023

@author: jonat
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G

# Taille de la grille de simulation
N = 2001


#indice du pixel central
s = np.floor(N/2)-1


Rg = 650


def gTran(rap,dN,a1,a2,bg):
    """
    Calcul le transit d'une exoplanete devant son étoile sur la grille de simulation.

    Parameters
    ----------
    rap : float
        rapport entre le rayon de la planete et de l'etoile.
    dN : int
        Deplacement de la planete entre les frames.
    a1 : float
        Coefficient du terme d'ordre 1 du "limb darkening".
    a2 : float
        Coefficient du terme d'ordre 2 du "limb darkening".
    bg : int
        Position en y du centre de la planete par rapport au centre de l'etoile.

    Returns
    -------
    In : array
        Intensite normalisee de l'etoile avec l'exoplanete.
    tg : array
        Frames de simulation.

    """
    
    #rayon de la planete dans la grille
    rg = int(rap*Rg)
    
    #rayons au carré
    Rgsq = Rg**2
    rgsq = rg**2
    
    #position du centre de la planete dans la grille
    py = s + bg
    
    
    #nombre de frames dans la simulation
    Tg = int((N-2*rg)/dN)
    
    #tableau des frames de la simulation
    tg = np.arange(Tg)
    
    #calcul du terme d'ordre 0 du "limb darkening"
    a0 = 1-a1-a2
    
    
    #creation de la grille de position des pixels
    [posy,posx] = np.indices((N,N))
    
    #position initiale du centre de la planete
    p0x = rg-1
    
    #calcul de la distance carré des pixels par rapport au centre de l'etoile
    Dsq = (posx-s)**2 + (posy-s)**2
    
    #determination des pixels appartenant à l'etoile
    instar = np.where((Dsq<=Rgsq),1,0)
    
    #calcul des distance carré des pixels avec le centre de la planete à la premiere frame
    dsq = (p0x-posx)**2 + (posy-py)**2
    
    #determination des pixels appartenant à la planete initialement
    inpy,inpx0, = np.where((dsq<=rgsq))
    
    #calcul du nombre de pixel dans la planete
    inpxs = inpx0.size
    
    #calculer la position de tout les pixels se trouvant dans la planete pour chaque frame
    inpxL = np.tile(inpx0,Tg) + dN*np.repeat(tg,inpxs)
    
    inpyL = np.tile(inpy,Tg)
    
    
    #calcul de l'intensite pour chaque pixel avec le "limb darkening"
    cos = np.sqrt(instar*(1-(Dsq/Rgsq)))
    i0 = a0*instar + a1*cos + a2*cos**2
    
    #extraire les intensites des pixels qui caché par la planete pour chaque frame
    Im = i0[inpyL,inpxL]
    
    #calculer l'intensite totale bloqué par la planete pour chaque frame
    I1 = np.resize(Im,(Tg,inpxs)).sum(axis=1)
    
    
    #calculer l'intensite totale de l'etoile seule
    I0 = i0.sum()
    
    #calculer l'intensité totale avec la planete pour chaque frame
    I = I0 - I1
    
    #normaliser l'intensite
    In = I/I0
    
    return In,tg

def Tran(M,R_s,rap,R,a1,a2,b,dN=1):
    """
    Calcul le transit d'une exoplanete et affiche la courbe de transit.

    Parameters
    ----------
    M : float
        Masse de l'etoile en kg.
    R_s : float
        Rayon de l'etoile en m.
    rap : float
        Rapport entre le rayon de la planete et de l'etoile.
    R : float
        Rayon de l'orbite de la planete en m.
    a1 : float
        Coefficient du terme d'ordre 1 du "limb darkening".
    a2 : float
        Coefficient du terme d'ordre 2 du "limb darkening".
    b : float
        Parametre d'impact.
    dN : int
        Deplacement de la planete entre les frames.

    Returns
    -------
    None.

    """
    
    #calculer la vitesse orbitale de la planete
    v = np.sqrt(G*M/R)
    
    #determiner l'intervalle de temps en sec. entre chaque frame
    dt = dN*R_s/(v*Rg)
    
    #mettre l'intervalle de temps en h
    dT = dt/3600
    
    
    #calculer l'inclinaison de l'orbite de la planete
    i = np.degrees(np.arccos(b*R_s/R))
    
    #calculer la hauteur du centre de la planete par rapport au centre de l'étoile dans la grille de simulation
    bg = b*Rg
    
    #extraire le tableau des intensités et des frames de la fonction de simulation
    In,tg = gTran(rap,dN,a1,a2,bg)
    
    #transformer les no de frames en heures
    t = tg*dT
    
    #afficher la courbe de transit
    plt.figure()
    plt.plot(t,In)
    plt.xlabel("t (h)")
    plt.ylabel("Flux")
    plt.show()
    
    #determiner et donner le temps réel du transit
    tind, = np.where(In!=1.00)
    print(f"t_exp = {t[tind[-1]]-t[tind[0]]:.2f} h","\n")
    
    #determiner et donner le temps théorique du transit si i=90°
    print(f"t_th = {2*R_s*(1 + rap)*np.sqrt(R/(G*M))/3600:.2f} h","\n")
    
    #donner l'inclinaison de l'orbite
    print(f"i = {np.floor(i):1.0f}° {np.floor((i-np.floor(i))*60):1.0f}' {np.floor(i*3600-np.floor(i*60)*60):1.0f}''","\n")
    
    print(f"En utilisant un déplacement de l'exoplanète de {dN} pixels entre chaque image, nous obtenons un intervalle de {dT:.2f} h entre chaque image.")
    
#rapport entre le rayon de Jupiter et du Soleil   
rsj = 6.9911e7/6.9634e8

#rapport entre le rayon de la Terre et du Soleil


Tran(1.99e30,6.9634e8,rsj,7.78e11,0.93,-0.23,0)



