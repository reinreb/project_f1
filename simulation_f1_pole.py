#Simulation F1 Pole lap time 
#Evan Bernier, 16/01/2026

#---------------IMPORT--------------
import pandas as pd
import numpy as np
import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
from fastf1.core import Laps
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

cache_dir = Path(__file__).parent.resolve() / "f1_cache"
cache_dir.mkdir(parents=True, exist_ok=True)  # crée le dossier si absent
fastf1.Cache.enable_cache(str(cache_dir))     # activer le cache
print("Cache dir:", cache_dir, "| exists:", cache_dir.exists(), "| is_dir:", cache_dir.is_dir())

"""
Circuit = "Monza"
Seasons = 2018 -> 2025
Sessions = FP1, FP2, FP3 and Qualifying
Weather = dry sessions only
Era = 2018 -> 2021 and 2022 -> 2025
Model = LinearRegression
"""
#---------------CONFIG--------------

track = "Italian Grand Prix"
seasons = range(2018, 2025)
dry_obligation = {"rain_fraction_max": 0.0}
sessions_studied = ["FP1", "FP2", "FP3", "Qualifying"]


SESSION_MAP = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Qualifying": "Qualifying"
}

def available_session_names(year: int, event_name: str):
    """Liste des noms de sessions disponibles pour cet événement"""
    sched = fastf1.get_event_schedule(year, include_testing=False)
    row = sched.loc[sched["EventName"] == event_name]
    if row.empty:
        return []
    # Colonnes Session1..Session5
    session_cols = [c for c in row.columns if c.startswith("Session") and not c.endswith("Date")]
    names = []
    for c in session_cols:
        val = row.iloc[0][c]
        if isinstance(val, str) and len(val):
            names.append(val)
    return names

def canonical_session_name(year: int, event_name: str, short_name: str):
    """Retourne le nom officiel existant pour short_name (FP1/FP2/FP3/Qualifying), sinon None"""
    cand = SESSION_MAP.get(short_name, short_name)
    avail = available_session_names(year, event_name)
    if cand in avail:
        return cand
    else:
        return None

#---------------DATA FROM FASTF1--------------

#retourne un dictionnaire avec pour chaque saison, les FL de chaque session du circuit 
def get_events(s, t):
    list_id_event = []
    for year in s :
        event = fastf1.get_event(year, t)
        nb_round = int(event.RoundNumber)
        list_id_event.append((int(year), nb_round, t))
    return list_id_event

list_ids = get_events(seasons, track)

#retourne true si la session "session_name" a connu de la pluie => si oui alors on ne la considere pas pour l'apprentissage
def rain(year, event_name, session_name):
    sess_name = canonical_session_name(year, event_name, session_name)
    if not sess_name:
        return False
    s = fastf1.get_session(year, event_name, session_name)
    s.load(laps=False, weather=True)
    weather_data = s.weather_data
    if weather_data is None or "Rainfall" not in weather_data.columns:
        return False
    rain_column = weather_data["Rainfall"]
    rained = bool(rain_column.fillna(False).astype(bool).any())
    return rained


def load_FL_from_session(year, s):
    sess_name = canonical_session_name(year, track, s)
    if not sess_name:
        return None
    session = fastf1.get_session(year, track, s)
    session.load(laps=True, weather=True)
    laps = session.laps
    if laps is None or laps.empty:
        return None
    #filtrer tours invalides si colonnes présentes
    if "Deleted" in laps.columns:
        laps = laps[~laps["Deleted"]]
    if "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"]]
    #colonnes requises
    if laps.empty or ("LapTime" not in laps.columns) or ("Driver" not in laps.columns):
        return None
    list_fastest_laps = []
    drivers = pd.unique(laps["Driver"])
    for drv in drivers:
        drvs_laps = laps.pick_drivers(drv)
        #sauter les pilotes sans tour exploitable
        if drvs_laps is None or drvs_laps.empty:
            continue
        if "LapTime" not in drvs_laps.columns:
            continue
        drvs_laps = drvs_laps.dropna(subset=["LapTime"])
        if drvs_laps.empty:
            continue
        #meilleur tour du pilote
        drvs_fastest_lap = drvs_laps.pick_fastest()
        #ignorer si le LapTime est NaN
        if ("LapTime" in drvs_fastest_lap.index) and pd.notna(drvs_fastest_lap["LapTime"]):
            list_fastest_laps.append(drvs_fastest_lap)

    if not list_fastest_laps:
        td = laps["LapTime"].min()
        if pd.notna(td):
            fast_lap_time_s = float(td.total_seconds())
        else:
            fast_lap_time_s = None        
        print(f"{year} {track} {s} best lap (fallback): {fast_lap_time_s} s")
        return fast_lap_time_s
    fastest_df = pd.DataFrame(list_fastest_laps)
    fastest_laps = Laps(fastest_df).sort_values(by="LapTime").reset_index(drop=True)
    td = fastest_laps.pick_fastest().LapTime
    if pd.notna(td):
        fast_lap_time_s = float(td.total_seconds())
    else:
        fast_lap_time_s = None
    print(f"{year} {track} {s} best lap: {fast_lap_time_s} s")
    return fast_lap_time_s

load_FL_from_session(max(seasons), "FP2")


def q3_pole_laptime(year):
    try:
        sess = fastf1.get_session(year, track, "Qualifying")
        sess.load(laps=False, weather=True)
        res = sess.results
    except Exception as e:
        return None
    if res is not None and "Q3" in res.columns:
        td = res["Q3"].dropna().min()
        if pd.notna(td):
            return float(td.total_seconds())
        else:
            return None
            
#dict des seances de FP avec pluie ou non et des Q3 sauf celle de l'année étudiée
def FL_data_gathered(year_of_analyse):
    dict_FL = {}
    for i in seasons:
        rec = {}
        for s in ("FP1", "FP2", "FP3"):
            rec[f"{s.lower()}_best"] = load_FL_from_session(i, s)
            rec[f"{s.lower()}_rain"] = rain(i, track, s)
        if i != year_of_analyse:
            rec["pole_q3"] = q3_pole_laptime(i)
            rec["quali_rain"] = rain(i, track, "Qualifying")
        else:
            rec["pole_q3"] = None
            rec["quali_rain"] = None
        dict_FL[i] = rec
    return dict_FL

#dict {year : {fp1 best : .., fp1 rain : .., }, ...}
#print(FL_data_gathered(year_of_analyse=2025))

#on passe à un dataframe exploitable 
def dataframe_exp(year_of_analyse):
    dict_FL = FL_data_gathered(year_of_analyse)
    dataframe = pd.DataFrame.from_dict(dict_FL, orient='index') #on fait un dataframe (matrice) avec en ligne les annees et en colonne les sessions (->FL) et la pluie pour chaque session (->bool)
    #print(dataframe.head())
    return dataframe

dataframe_exp(2025)

#clean des données pour preparation à l'entrainement
def clean(year_of_analyse):
    df = dataframe_exp(year_of_analyse)
    cols_essentielles = ['pole_q3', 'fp1_best', 'fp2_best', 'fp3_best']
    df_clean = df.dropna(subset=cols_essentielles).copy()
    for col in ['fp1_rain', 'fp2_rain', 'fp3_rain', 'quali_rain']:
        df_clean[col] = df_clean[col].fillna(False)
    cols = ['fp1_best', 'fp2_best', 'fp3_best', 'fp1_rain', 'fp2_rain', 'fp3_rain', 'quali_rain']
    X = df_clean[cols]
    Y = df_clean['pole_q3']
    #print(X.head())
    #print("y :", Y.tolist())
    return X, Y
    
X, Y = clean(2025)

#entrainement du modele : création de deux listes (une pour l'entrainement (anciennes années), une autre pour les tests (années recentes))

def arrays_train_test(year_of_analyse):
    train_years = []
    test_years = []
    df = dataframe_exp(year_of_analyse)
    cols_essentielles = ['pole_q3', 'fp1_best', 'fp2_best', 'fp3_best']
    df = df.dropna(subset=cols_essentielles).copy()
    for year in df.index:
        if year <= 2022:
            train_years.append(year)
    for year in df.index:
        if year > 2022:
            test_years.append(year)
    cols = ['fp1_best', 'fp2_best', 'fp3_best', 'fp1_rain', 'fp2_rain', 'fp3_rain', 'quali_rain']
    for c in cols:
        if c.endswith('_rain'):
            df[c] = df[c].fillna(False)
    X = df[cols]
    Y = df['pole_q3']
    #on selectionne uniquement ce qui nous interesse : train_years et test_years grace à loc
    X_train = X.loc[train_years]
    Y_train = Y.loc[train_years]
    X_test = X.loc[test_years]
    Y_test = Y.loc[test_years]
    return X_train, Y_train, X_test, Y_test, test_years

X_train, Y_train, X_test, Y_test, test_years = arrays_train_test(2025)

#regression lineaire
def linear_reg():
    model = LinearRegression()
    model.fit(X_train, Y_train)
    #predictions
    Y_prediction_train = model.predict(X_train)
    Y_prediction_test = model.predict(X_test)
    #calcul de l'erreur abs moy => diff entre la prediction et la realité
    err_train = mean_absolute_error(Y_train, Y_prediction_train)
    err_test = mean_absolute_error(Y_test, Y_prediction_test)
    #calcul de l'erreur quadratique moy racine 
    err_racine_test = np.sqrt(mean_squared_error(Y_test, Y_prediction_test))
    print("Linear Regression")
    print(f"Err abs moy Train : {err_train:.3f} secondes")
    print(f"Err abs moy  Test  : {err_test:.3f} secondes")
    print(f"RMSE Test : {err_racine_test:.3f} secondes")
    print("\nPrédictions vs Réalité (test) :")
    for i, year in enumerate(test_years):
        print(f"{year} : real={Y_test.iloc[i]:.3f}s, pred={Y_prediction_test[i]:.3f}s")

linear_reg()

def linear_reg_monza_2024():
    print("-------------------------------")
    print("Linear Regression Monza 2024 Q3 ")
    model_final = LinearRegression()
    model_final.fit(X_train, Y_train)
    X_real_2024 = X.loc[[2024]]
    print("X real 2025 : ", X_real_2024)
    #predictions
    pred_q3_2024 = model_final.predict(X_real_2024)[0]
    real_q3_2024 = q3_pole_laptime(2024)
    if real_q3_2024:
        erreur = abs(pred_q3_2024 - real_q3_2024)
        print(f"Q3 RÉEL 2024       : {real_q3_2024:.3f} secondes")
        print(f"Q3 PRED 2024       : {pred_q3_2024:.3f} secondes")
        print(f"ERREUR ABSOLUE     : {erreur:.3f} secondes")
        print(f"ECART RELATIF      : {(erreur/real_q3_2024)*100:.2f}%")
    return pred_q3_2024

linear_reg_monza_2024()
