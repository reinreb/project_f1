"""
Auteur : Evan Bernier
But : affichage de boites à moustaches pour chaque pilote le plus rapide de chaque écurie (race pace)
"""

import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


import fastf1
import fastf1.plotting

def format_laptime(seconds):
    sec = seconds % 60
    return f"{sec:05.2f}"

fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='white')

year = 2026
track = "Chinese Grand Prix"
session = 'R'

#Chargement de la session étudié (course)
race = fastf1.get_session(2026, track, session)
race.load()

#Chargement de tout les tours de tous les pilotes => on filtre les tours "slow" et on ne considere que les tours "rapides"
driver_laps = race.laps.pick_drivers(race.drivers[:20]).pick_quicklaps()
driver_laps = driver_laps.reset_index() #remet à plat les index des lignes après filtrage pour driver_laps


#Création de l'affichage 
aff, ax = plt.subplots(figsize=(14, 7)) #15 de longueur et 7 de largeur (en inches)

#Conversion les "LapTime" en secondes à cause de Seaborn 
driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
driver_laps["LapTime_str"] = driver_laps["LapTime(s)"].apply(format_laptime)

mean_laptimes = driver_laps.groupby("Driver")["LapTime(s)"].mean()
sorted_drivers = mean_laptimes.sort_values().index.tolist()[:10]

#Création des boites à moustaches 
box = sb.boxplot(data=driver_laps,
            x="Driver", #sur les abscisses 
            y="LapTime(s)", #sur les ordonnées 
            hue="Driver", #on prend une couleur pour chaque pilote (pour chaque BaM) => Seaborn spérare les données pour chaque pilote
            order=sorted_drivers, 
            palette=fastf1.plotting.get_driver_color_mapping(session=race), #on recupere les color deja founi par fastf1
            showmeans = True,
            legend=False,
            meanprops={
                "marker": ".",               
                "markerfacecolor": "black", #couleur à l'intérieur du marker 
                "markeredgecolor": "black", #couleur du bord du marker
                "markersize": 7
                }       
)


def calcul_gap_with_p1(i):
    if i == 0:
        return ""
    gap_seconds = mean_laptimes[sorted_drivers[i]] - mean_laptimes[sorted_drivers[0]]
    gap_str = f"{gap_seconds:.2f}"                     
    return f" (+{gap_str})"

for i, driver in enumerate(sorted_drivers):
    mean_val = mean_laptimes[driver]
    ax.text(
        i,                          
        driver_laps["LapTime(s)"].min() - 0.20,  # position y sous les boxplots
        format_laptime(mean_val) + calcul_gap_with_p1(i),          # texte affiché
        ha='center',
        fontsize=11,
        fontstyle='oblique'
    )

#Légende de l'affichage 

ax.set_ylabel("Laptime(s)")
legend_elements = [
    Line2D([0], [0], color='black', lw=1, label='Median'),
    Line2D([0], [0], marker='.', linestyle='None', color='black', markerfacecolor='black', markersize=7, label='Average')]
ax.legend(handles=legend_elements)
plt.suptitle(f'{year} F1 {track} Race Pace\n') #Titre principal 
sb.despine(left=True, bottom=True) #on supprime les lignes au bord de l'affichage 

plt.tight_layout()
plt.show()
