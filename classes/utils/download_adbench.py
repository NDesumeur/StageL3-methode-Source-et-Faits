import os
import urllib.request
import urllib.error
import urllib.parse
import json

def telecharger_adbench_classical():
    """Télécharge l'intégralité des datasets tabulaires classiques depuis le dépôt GitHub ADBench."""
    dossier_cible = os.path.join(os.getcwd(), "data", "adbench")
    os.makedirs(dossier_cible, exist_ok=True)

    print(" Recherche des datasets ADBench Classical (sur GitHub API)...")
    
    api_url = "https://api.github.com/repos/Minqi824/ADBench/contents/adbench/datasets/Classical"
    
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            fichiers_json = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f" Impossible de contacter l'API GitHub : {e}")
        return

    noms_fichiers = [f['name'] for f in fichiers_json if f['name'].endswith('.npz')]
    total = len(noms_fichiers)
    print(f" {total} datasets trouvés. Début du téléchargement vers {dossier_cible} ...")

    for i, nom in enumerate(noms_fichiers):
        chemin_sauvegarde = os.path.join(dossier_cible, nom)
        
        if os.path.exists(chemin_sauvegarde):
            continue
            
        url_brute = f"https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{urllib.parse.quote(nom)}"
        
        print(f" Téléchargement de {nom} ({i+1}/{total})")
        try:
            urllib.request.urlretrieve(url_brute, chemin_sauvegarde)
        except Exception as e:
            print(f"    Erreur: {e}")
            
    print(f" Opération terminée. Les fichiers \".npz\" sont dans data/adbench/.")

if __name__ == '__main__':
    telecharger_adbench_classical()
