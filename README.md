# MachineLearningMail
Rangement automatique de mails

Pour lancer le programme il suffit de lancer la commande :
python3 mail2directory.py

Il est possible de modifier plusieurs paramètres de ce programme :

1) le dataset
Le ou les dataset sur lesquels travailera le programme peuvent être modifié directement dans la variable : datasets du fichier mail2directory.py.
A noter qu'il y a plusieurs dataset disponible : all1, all2, all3, kaminski1, kaminski2 qui sont présenté dans le rapport

2) le descripteur
Pour modifier la partie descripteurs, il faut décommenter les descripteurs que l'ont veut lancer dans la partie # Descriptors

3) le model
Pour modifier le model, il faut décommenter le model que vous souhaitez utiliser dans la partie # Train classification model du fichier mail2directory.py. Les modeles et leurs paramètres sont définis dans le dossier mail2directory.py
Attention 


Un second programme peut être lancé pour tester les meilleurs paramètres pour la partie descripteurs. Celui-ci peut être lancé avec la commande :
python3 searchBestDesc.py
Le ou les dataset sur lesquels travailera le programme peuvent être modifié directement dans la variable : datasets du fichier searchBestDesc.py.