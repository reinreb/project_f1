"""
Auteur : Evan Bernier
But : affichage de boites à moustaches pour chaque pilote le plus rapide de chaque écurie (race pace)
"""

import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt

import fastf1
import fastf1.plotting

fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='white')


#chargement de la session étudié (course)
race = fastf1.get_session(2025, "Brazil", 'R')
race.load()

#chargement de tout les tours de tous les pilotes => on filtre les tours "slow" et on ne considere que les tours "rapides"
driver_laps = race.laps.pick_drivers(race.drivers[:20]).pick_quicklaps()
driver_laps = driver_laps.reset_index() #remet à plat les index des lignes après filtrage pour driver_laps


#recuperation des abreviations de chaque pilote pour ensuite les inserer dans la liste finishing_order
finishing_order = []
for i in race.drivers[:20]:
    abbreviation = race.get_driver(i)["Abbreviation"]
    finishing_order.append(abbreviation)

#création de l'affichage 
aff, ax = plt.subplots(figsize=(15, 7)) #15 de longueur et 7 de largeur (en inches)

#conversion les "LapTime" en secondes à cause de Seaborn 
driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

#création des boites à moustaches 
sb.boxplot(data=driver_laps,
            x="Driver", #sur les abscisses 
            y="LapTime(s)", #sur les ordonnées 
            hue="Driver", #on prend une couleur pour chaque pilote (pour chaque BaM) => Seaborn spérare les données pour chaque pilote
            order=finishing_order, 
            palette=fastf1.plotting.get_driver_color_mapping(session=race), #on recupere les color deja founi par fastf1
            showmeans = True,
            meanprops={
                "marker": ".",               
                "markerfacecolor": "black", #couleur à l'intérieur du marker 
                "markeredgecolor": "black", #couleur du bord du marker
                "markersize": 10
                }       
)

#legende de l'affichage 
ax.set_xlabel("Pilote") 
ax.set_ylabel("Temps au tour")
plt.suptitle("2025 Brazil : Rythme de course (Race pace)") #titre principal 
sb.despine(left=True, bottom=True) #on supprime les lignes au bord de l'affichage 

plt.tight_layout()
plt.show()
