# Real-Time-Project-Azure

## Contenu
- [Introduction à l'architecture et approche globale](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#1-introduction-%C3%A0-larchitecture-et-approche-globale);
- [Architecture technique détaillée](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#2-architecture-technique-d%C3%A9taill%C3%A9e);
- [Création des services Event Hub et Azure Stream Analytics puis liaison des services](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#3-cr%C3%A9ation-des-services-event-hub-et-azure-stream-analytics-puis-liaison-des-services);
- [Développement du générateur et envoie de données du générateur vers le service Event Hub ](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#4-d%C3%A9veloppement-du-g%C3%A9n%C3%A9rateur-et-envoie-de-donn%C3%A9es-du-g%C3%A9n%C3%A9rateur-vers-le-service-event-hub);
- [Mise en place d’une Azure Cosmos DB puis liaison avec Stream Analytics et création d’un historique de données](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#5-mise-en-place-dune-azure-cosmos-db-puis-liaison-avec-stream-analytics-et-cr%C3%A9ation-dun-historique-de-donn%C3%A9es);
- [Développement de l’algorithme de ML avec Spark et enregistrement du modèle sous Azure ML](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#6-d%C3%A9veloppement-de-lalgorithme-de-ml-avec-spark-et-enregistrement-du-mod%C3%A8le-sous-azure-ml);
- [Déploiement du modèle dans un conteneur Azure Kubernetes](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#7-d%C3%A9ploiement-du-mod%C3%A8le-dans-un-conteneur-azure-kubernetes);
- [Mise en place d’un Event Hub / Azure Stream Analytics / Azure Cosmos DB pour l’architecture finale en temps réel](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#8-mise-en-place-dun-event-hub--azure-stream-analytics--azure-cosmos-db-pour-larchitecture-finale-en-temps-r%C3%A9el);
- [Modification de la requete SQL Stream Analytics pour scorer les données](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#9-modification-de-la-requete-sql-stream-analytics-pour-scorer-les-donn%C3%A9es);
- [Création et intégration des deux dashboards Power BI](https://github.com/Katalyse/Real-Time-Project-Azure/blob/main/README.md#10-cr%C3%A9ation-et-int%C3%A9gration-des-deux-dashboards-power-bi);


# 1. Introduction à l'architecture et approche globale

<p align="justify">
L’objectif de ce projet est de créer une architecture cloud qui permet de gérer un scénario d’analyse de données en temps réel. Plus précisément, nous souhaitons instantanément savoir si une transaction bancaire est frauduleuse ou non. Il faut pour cela s’appuyer sur une architecture qui va permettre de stocker, d’analyser et de restituer les données le plus rapidement possible.
</p>

<p align="justify">
Pour ce projet, nous disposons d’un générateur de données en local. Il est donc chargé de générer des données concernant des transactions bancaires et d’envoyer des requêtes contenant ces données vers le service Event Hub. Dans la première partie de cette architecture, on va créer un historique des transactions pour pouvoir créer un modèle de machine learning. Le service Event Hub capture les événements provenant du générateur contenant des données labelisées pour savoir si la transaction est frauduleuse ou non. Les données sont transmises vers le service Azure Stream Analytics qui redirige les données vers une base de données Azure Cosmos DB permettant de stocker des documents au format JSON. Une fois que l’on dispose de plusieurs dizaines de milliers d’enregistrement, on va pouvoir créer notre modèle de ML. On utilisera un environnement Spark sur Azure Synapse Analytics pour réaliser cette tâche. L’utilisation d’un cluster Spark permet de créer des algorithmes de ML à partir d’un grand volume de données de manière rapide. Une fois que le modèle est construit, il est enregistré dans Azure Machine Learning pour qu’il soit déployable facilement. On va créer un conteneur Azure Kubernetes pour héberger le modèle. Une fois que l’on dispose du modèle dans un conteneur, on peut mettre en place la deuxième partie de l’architecture qui va permettre le traitement des données en temps réel. 
</p>

<p align="justify">
Le générateur va envoyer un certain nombre de données vers le service Event Hub. Cette fois-ci, les données ne sont pas labelisées et l’algorithme doit prédire si la transaction est frauduleuse ou non. Une architecture lambda est mise en place au niveau du service de streaming d’Azure. En effet, le service Event Hub est chargé de recueillir les événements envoyés depuis le générateur puis va transmettre ces événements vers le service Azure Stream Analytics. Dans un premier temps, ce service envoie les données vers le conteneur Kubernetes pour scorer les données à l’aide de l’algorithme de machine learning. Le service Azure Stream Analytics récupère cette prédiction puis envoie les données avec la prédiction à la fois vers Power BI et vers Azure Cosmos DB. Le rapport Power Bi permet d’obtenir un visuel en temps réel pour repérer des transactions frauduleuses et Azure Cosmos DB permet de stocker toutes ces données. Enfin, un autre rapport Power BI est créé à partir de l’historique des transactions stockées dans Azure Cosmos DB.
</p>

<p align="justify">
Voici l’architecture mise en place ainsi que la liste des étapes dans l’ordre chronologique pour mettre ne place cette architecture :
</p>

![](/Pictures/archi.jpeg?raw=true)

# 2. Architecture technique détaillée
# 3. Création des services Event Hub et Azure Stream Analytics puis liaison des services
# 4. Développement du générateur et envoie de données du générateur vers le service Event Hub 
# 5. Mise en place d’une Azure Cosmos DB puis liaison avec Stream Analytics et Création d’un historique de données
# 6. Développement de l’algorithme de ML avec Spark et enregistrement du modèle sous Azure ML
# 7. Déploiement du modèle dans un conteneur Azure Kubernetes
# 8. Mise en place d’un Event Hub / Azure Stream Analytics / Azure Cosmos DB pour l’architecture finale en temps réel
# 9. Modification de la requete SQL Stream Analytics pour scorer les données
# 10. Création et intégration des deux dashboards Power BI



![](/Pictures/iRobotArchitecture.png?raw=true)


```Shell
az iot hub device-identity create -n <IOT_HUB_NAME> -d AIRobot1 --pd AIRobotEdge
```
