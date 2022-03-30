# Real-Time-Project-Azure

## Contenu
- [Introduction à l'architecture et approche globale](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Architecture technique détaillée](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Création des services Event Hub et Azure Stream Analytics puis liaison des services](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Développement du générateur et envoie de données du générateur vers le service Event Hub ](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Mise en place d’une Azure Cosmos DB / Liaison avec Stream Analytics / Test / Création d’un historique de données](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Développement de l’algorithme de ML avec Spark et enregistrement du modèle sous Azure ML](https://github.com/fredgis/AIRobot#azure-iot-edge-as-transparent-gateway);
- [Déploiement du modèle dans un conteneur Azure Kubernetes](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Mise en place d’un Event Hub / Azure Stream Analytics / Azure Cosmos DB pour l’architecture finale en temps réel](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Modification de la requete SQL Stream Analytics pour scorer les données](https://github.com/Katalyse/Real-Time-Project-Azure);
- [Création et intégration des deux dashboards Power BI](https://github.com/Katalyse/Real-Time-Project-Azure);


# 1. Introduction à l'architecture et approche globale

Le schéma d'architecture ci-après présente l'approche globable d'architecture.
![](/Pictures/archi.jpeg?raw=true)

# 2. Architecture technique détaillée
# 3. Création des services Event Hub et Azure Stream Analytics puis liaison des services
# 4. Développement du générateur et envoie de données du générateur vers le service Event Hub 
# 5. Mise en place d’une Azure Cosmos DB / Liaison avec Stream Analytics / Test / Création d’un historique de données
# 6. Développement de l’algorithme de ML avec Spark et enregistrement du modèle sous Azure ML
# 7. Déploiement du modèle dans un conteneur Azure Kubernetes
# 8. Mise en place d’un Event Hub / Azure Stream Analytics / Azure Cosmos DB pour l’architecture finale en temps réel
# 9. Modification de la requete SQL Stream Analytics pour scorer les données
# 10. Création et intégration des deux dashboards Power BI



![](/Pictures/iRobotArchitecture.png?raw=true)


```Shell
az iot hub device-identity create -n <IOT_HUB_NAME> -d AIRobot1 --pd AIRobotEdge
```
