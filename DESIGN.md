L'ancien dataloader, du DALI naïf, est appelé legacy dans ce document.
Ce projet vise à créer un dataloader optimisé pour du HPC. Il tire parti des observations suivantes sur les librairies open source disponibles.

# DALI

- +:
    - fusion d'opérateur,
    - optimisation des opérateurs de lecture de webdataset (cache, ...),
    - pipeline global de l'IO à la donnée pré-traitée -> haute perf.
- -:
    - peu flexible, graphe de calcul statique
        - fonctionnalité expérimentale avec définition dynamique du graphe, utilisé dans dino_loader.experimental.dynamic_pipeline.py
    - nécessite de définir les opérateurs d'augmentation avec les opérations DALI
        - visible dans le dataloader legacy utilisé actuellement sur Iliad
        - autre exemple dans dino_loader.pipeline._build_dinov2_pipeline
    - au début du travail, DALI ne supportait pas un pipeline full-CPU -> incompatible avec la CI sans créer du deadcode inutile
        - encore visible ici dans dino_loader.backends.cpu
    - le reader de webdataset n'est pas conçu pour du curriculum learning
        - avis des développeurs: https://github.com/NVIDIA/DALI/issues/6274 

# Webdataset (la lib python)

- +:
    - feature complete
    - compatible avec du pytorch habituel
- -:
    - full CPU/pas optimisé

# Le mix naïf

- utiliser webdataset pour réaliser le curriculum learning (exemple dans dino_loader.sources.wds_source.py)
- passer à DALI les données mixées et réaliser le décodage/augmentation sur GPU avec les opérateurs fusionnés

C'est fait dans ce projet afin de donner une référence si on souhaite maintenir une implémentation ultra-simple. Il "suffit" d'extraire le code de wds_source et les kernels DALI pour l'augmentation.

# Le mix optimisé

- essayer de répliquer les performances du reader de DALI en tenant compte des besoins de curriculum learning (dino_loader.sources.hpc_source.py)

# Le reste de ce repo
De la déco, typiquement pour permettre le checkpointing, proposer le masking iBot, ...
Il y a une forme de dépendance aux DatasetSpec que peut générer la lib. dino_datasets. 
Il est principalement vibecodé pour essayer d'aller vite: il y a des incohérences, des commentaires peu utiles, ...