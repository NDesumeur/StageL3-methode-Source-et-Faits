import os
import urllib.request
import json
import time
import urllib.parse

def telecharger_tout_adbench():
    """
    Télécharge tous les datasets tabulaires (.npz) du dépôt original ADBench (Minqi824).
    Cela inclut les dossiers Classical, CV_by_ResNet18, CV_by_ViT, NLP_by_BERT, NLP_by_RoBERTa.
    """
    
    # URL de base de l'API GitHub pour le dossier datasets
    api_url = "https://api.github.com/repos/Minqi824/ADBench/contents/adbench/datasets"
    
    # Dossier où stocker les données
    dossier_base = os.path.join(os.getcwd(), 'data', 'adbench')
    os.makedirs(dossier_base, exist_ok=True)
    
    print(f" Sauvegarde des datasets dans : {dossier_base}")
    print(" Recherche des dossiers ADBench...")
    
    # Requête pour obtenir la liste des dossiers
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            contenu = json.loads(response.read().decode('utf-8'))
            dossiers = [item['name'] for item in contenu if item['type'] == 'dir']
    except Exception as e:
        print(f" Erreur lors de l'accès à l'API GitHub : {e}")
        return

    print(f" {len(dossiers)} dossiers trouvés : {', '.join(dossiers)}")
    
    for dossier in dossiers:
        dossier_cible = os.path.join(dossier_base, dossier)
        os.makedirs(dossier_cible, exist_ok=True)
        
        print(f"\n Traitement du dossier: {dossier}")
        dossier_url = f"{api_url}/{urllib.parse.quote(dossier)}"
        
        try:
             req = urllib.request.Request(dossier_url, headers={'User-Agent': 'Mozilla/5.0'})
             with urllib.request.urlopen(req) as response:
                 fichiers_json = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f" Impossible de lister les fichiers du dossier {dossier}: {e}")
            continue

        fichiers_npz = [f for f in fichiers_json if f['name'].endswith('.npz')]
        total = len(fichiers_npz)
        print(f" {total} fichiers .npz trouvés dans {dossier}.")
    
        # Télécharger chaque fichier
        for i, fichier in enumerate(fichiers_npz, 1):
            nom_fichier = fichier['name']
            url_telechargement = fichier.get('download_url')
            if not url_telechargement:
                continue
                
            chemin_sauvegarde = os.path.join(dossier_cible, nom_fichier)
            
            # Ne pas re-télécharger si le fichier existe
            if os.path.exists(chemin_sauvegarde):
                print(f"  [{i}/{total}]  Déjà téléchargé : {nom_fichier}")
                continue
                
            print(f"  [{i}/{total}]  Téléchargement : {nom_fichier} ...", end="", flush=True)
            
            try:
                # Ajout de headers pour simuler un navigateur et éviter le blocage
                req_dl = urllib.request.Request(
                    url_telechargement,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                
                with urllib.request.urlopen(req_dl) as reponse, open(chemin_sauvegarde, 'wb') as f_out:
                    f_out.write(reponse.read())
                print(" OK")
                
                # Petite pause pour éviter de se faire bloquer par GitHub
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Erreur : {e}")

    print("\n Téléchargement terminé !")

if __name__ == "__main__":
    telecharger_tout_adbench()
