# uv (Gestionnaire Python express)

C'est le remplaçantde `pip` et `venv`. Ça fait exactement la même chose, mais en beaucou plus rapide?Parfait pour installer de gros modules comme Scikit-Learn instantanément.

## Cheat Sheet Commandes

**1. L'installer sur le PC (une seule fois)**
```bash
pip install uv
```

**2. Créer l'environnement du projet**
```bash
uv venv
```
*(Ne pas oublier de l'activer après : `.venv\Scripts\activate` sous Windows !)*

**3. Installer des librairies**
On rajoute juste `uv` devant nos commandes habituelles :
```bash
uv pip install numpy pandas scikit-learn
```

**4. Installer tout un projet d'un coup (depuis github etc)**
```bash
uv pip install -r requirements.txt
```
