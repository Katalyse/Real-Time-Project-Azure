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

Le schéma d'architecture ci-après présente l'approche globable d'architecture.
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
