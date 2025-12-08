"""
generate_backup.py
Générateur de dataset professionnel pour l'analyse de pannes cloud
Version finale corrigée avec bonne indentation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from typing import List, Dict, Tuple
import json

class ProfessionalDatasetGenerator:
    """Générateur de dataset Big Data pour analyse de pannes cloud"""
    
    def __init__(self):
        # Configuration des services cloud
        self.services = {
            'AWS': {
                'subreddit': 'aws',
                'components': ['EC2', 'S3', 'Lambda', 'RDS', 'CloudFront', 'DynamoDB', 'API Gateway'],
                'regions': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'sa-east-1'],
                'keywords': ['instance', 'bucket', 'function', 'database', 'distribution', 'table']
            },
            'Azure': {
                'subreddit': 'azure', 
                'components': ['Virtual Machines', 'Blob Storage', 'Functions', 'SQL Database', 'Cosmos DB', 'AKS'],
                'regions': ['East US', 'West Europe', 'Southeast Asia', 'Brazil South', 'Central US'],
                'keywords': ['vm', 'storage', 'function', 'sql', 'cosmos', 'kubernetes']
            },
            'Google Cloud': {
                'subreddit': 'googlecloud',
                'components': ['Compute Engine', 'Cloud Storage', 'BigQuery', 'Cloud Functions', 'Cloud SQL', 'GKE'],
                'regions': ['us-central1', 'europe-west1', 'asia-southeast1', 'southamerica-east1'],
                'keywords': ['compute', 'storage', 'bigquery', 'function', 'database', 'kubernetes']
            },
            'GitHub': {
                'subreddit': 'github',
                'components': ['Actions', 'Pages', 'Packages', 'Codespaces', 'Copilot', 'API'],
                'regions': ['US', 'EU', 'APAC'],
                'keywords': ['workflow', 'page', 'package', 'codespace', 'copilot', 'api']
            }
        }
        
        # Utilisateurs réalistes
        self.users = [
            'sysadmin_tech', 'devops_specialist', 'cloud_architect', 
            'platform_engineer', 'sre_lead', 'cloud_security_analyst',
            'data_engineer_01', 'ml_infra', 'backend_developer_42'
        ]
        
        # Catégories d'urgence avec templates professionnels
        self.urgency_templates = {
            'CRITICAL': {
                'templates': [
                    "CRITICAL: {service} {component} completely down in {region}",
                    "MAJOR OUTAGE: {service} {component} services unavailable",
                    "PRODUCTION INCIDENT: {service} {component} failure in {region}",
                    "SERVICE DISRUPTION: {service} {component} outage affecting customers",
                    "SYSTEM FAILURE: {service} {component} not responding in {region}"
                ],
                'base_score': (150, 400),
                'base_comments': (25, 100)
            },
            'HIGH': {
                'templates': [
                    "Severe performance degradation: {service} {component} in {region}",
                    "Critical issues with {service} {component} services",
                    "{service} {component} experiencing major errors",
                    "Service degradation report: {service} {component}",
                    "High severity problem with {service} {component}"
                ],
                'base_score': (80, 200),
                'base_comments': (15, 60)
            },
            'MEDIUM': {
                'templates': [
                    "Performance issues with {service} {component}",
                    "{service} {component} intermittent failures",
                    "Partial service degradation: {service} {component}",
                    "Operational problems on {service} {component}",
                    "Service disruption: {service} {component} in {region}"
                ],
                'base_score': (30, 120),
                'base_comments': (8, 35)
            },
            'LOW': {
                'templates': [
                    "Discussion: {service} {component} configuration best practices",
                    "Question about {service} {component} implementation",
                    "Learning resources for {service} {component}",
                    "{service} {component} cost optimization strategies",
                    "Technical discussion: {service} {component} architecture"
                ],
                'base_score': (5, 50),
                'base_comments': (2, 20)
            }
        }
        
        # Patterns d'incidents temporels
        self.incident_patterns = {
            'rapid_onset': {'duration_min': 30, 'duration_max': 120, 'posts_per_hour': (15, 30)},
            'gradual_degradation': {'duration_min': 120, 'duration_max': 360, 'posts_per_hour': (8, 20)},
            'intermittent': {'duration_min': 180, 'duration_max': 480, 'posts_per_hour': (5, 15)}
        }
    
    def generate_incident(self, incident_id: int, pattern: str = 'rapid_onset') -> List[Dict]:
        """Génère une séquence cohérente de posts pour un incident"""
        incidents = []
        
        # Sélection aléatoire d'un service
        service_name = random.choice(list(self.services.keys()))
        service_config = self.services[service_name]
        
        # Configuration de l'incident
        pattern_config = self.incident_patterns[pattern]
        duration = random.randint(pattern_config['duration_min'], pattern_config['duration_max'])
        total_posts = int(duration / 60 * random.uniform(*pattern_config['posts_per_hour']))
        
        # Heure de début
        incident_start = datetime.now() - timedelta(
            days=random.randint(1, 7),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Composant et région affectés
        component = random.choice(service_config['components'])
        region = random.choice(service_config['regions'])
        
        # Génération de la timeline
        for post_num in range(total_posts):
            # Progression temporelle dans l'incident
            time_offset = timedelta(minutes=random.randint(0, duration))
            post_time = incident_start + time_offset
            
            # Détermination de l'urgence selon la progression
            if post_num < total_posts * 0.2:  # Phase initiale
                urgency = 'CRITICAL'
            elif post_num < total_posts * 0.6:  # Phase critique
                urgency = random.choices(['CRITICAL', 'HIGH'], weights=[0.3, 0.7])[0]
            else:  # Phase de résolution
                urgency = random.choices(['HIGH', 'MEDIUM'], weights=[0.4, 0.6])[0]
            
            # Génération du titre
            template = random.choice(self.urgency_templates[urgency]['templates'])
            title = template.format(
                service=service_name,
                component=component,
                region=region
            )
            
            # Calcul des métriques
            score_range = self.urgency_templates[urgency]['base_score']
            comments_range = self.urgency_templates[urgency]['base_comments']
            
            incidents.append({
                'incident_id': f'INC-{incident_id:03d}',
                'incident_start': incident_start,
                'incident_pattern': pattern,
                'post_sequence': post_num + 1,
                'service': service_name,
                'component': component,
                'region': region,
                'title': title,
                'urgency': urgency,
                'is_outage': 1,
                'timestamp': post_time,
                'score': random.randint(*score_range),
                'num_comments': random.randint(*comments_range),
                'total_incident_posts': total_posts
            })
        
        return incidents
    
    def generate_normal_activity(self, num_posts: int) -> List[Dict]:
        """Génère l'activité normale (non-incident)"""
        posts = []
        
        for i in range(num_posts):
            service_name = random.choice(list(self.services.keys()))
            service_config = self.services[service_name]
            
            # Timestamp aléatoire sur 30 jours
            post_time = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Sélection des métadonnées
            component = random.choice(service_config['components'])
            region = random.choice(service_config['regions'])
            
            # Génération du titre (toujours LOW urgency pour activité normale)
            template = random.choice(self.urgency_templates['LOW']['templates'])
            title = template.format(
                service=service_name,
                component=component,
                region=region
            )
            
            posts.append({
                'incident_id': None,
                'incident_start': None,
                'incident_pattern': None,
                'post_sequence': 0,
                'service': service_name,
                'component': component,
                'region': region,
                'title': title,
                'urgency': 'LOW',
                'is_outage': 0,
                'timestamp': post_time,
                'score': random.randint(5, 50),
                'num_comments': random.randint(0, 20),
                'total_incident_posts': 0
            })
        
        return posts
    
    def generate_text_content(self, row: Dict) -> str:
        """Génère un contenu textuel réaliste pour un post"""
        
        if row['urgency'] == 'CRITICAL':
            templates = [
                f"Our production systems relying on {row['service']} {row['component']} in {row['region']} are experiencing complete service interruption. All API calls returning 5xx errors with 100% failure rate. Monitoring alerts triggered across all environments. Team is actively investigating but requires immediate community feedback regarding scope of impact.",
                f"CRITICAL SERVICE OUTAGE: {row['service']} {row['component']} services are completely unavailable in {row['region']} region. Production workloads affected. Initial investigation suggests authentication service failures. Official status page has not been updated. Requesting information from other affected parties.",
                f"MAJOR PRODUCTION INCIDENT: Complete failure of {row['service']} {row['component']} infrastructure in {row['region']}. Services began failing approximately 15 minutes ago. Error logs show consistent timeout and connection refusal patterns. Service Level Objectives severely breached. Seeking confirmation from other organizations."
            ]
        
        elif row['urgency'] == 'HIGH':
            templates = [
                f"Severe performance degradation detected in {row['service']} {row['component']} services within {row['region']}. Failure rate currently at 65-70% with significant latency increase. Partial service availability but core functionality impacted. Engineering team investigating root cause.",
                f"OPERATIONAL CRISIS: {row['service']} {row['component']} experiencing critical performance issues in {row['region']}. Monitoring indicates 40-50% error rate across service endpoints. Customer impact is substantial but not complete. Technical teams engaged in triage and mitigation."
            ]
        
        elif row['urgency'] == 'MEDIUM':
            templates = [
                f"Intermittent issues reported with {row['service']} {row['component']} in {row['region']}. Error rate approximately 15-20% with sporadic timeout occurrences. Service degradation noticeable but core functionality remains operational. Investigation ongoing.",
                f"Partial service degradation: {row['service']} {row['component']} showing elevated error rates in {row['region']}. Monitoring indicates periodic failures affecting specific service endpoints. Impact is limited but requires attention."
            ]
        
        else:  # LOW
            templates = [
                f"Technical discussion regarding {row['service']} {row['component']} implementation patterns and best practices. Seeking community input on optimization strategies and architectural considerations.",
                f"Educational query about {row['service']} {row['component']} configuration and operational management. Looking for experienced practitioners to share insights and recommendations.",
                f"Cost optimization analysis for {row['service']} {row['component']} deployments. Discussion of resource allocation strategies and efficiency improvements."
            ]
        
        return random.choice(templates)
    
    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule des features temporelles pour l'analyse"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['days_since_epoch'] = (df['timestamp'] - pd.Timestamp('2024-01-01')).dt.days
        
        # Feature: densité de posts par fenêtre temporelle
        df = df.sort_values('timestamp')
        df['post_density_1h'] = df.rolling('1h', on='timestamp')['id'].count().values
        
        return df
    
    def create_dataset(self, total_size: int = 800, incident_ratio: float = 0.15) -> pd.DataFrame:
        """Crée le dataset complet avec contrôle de qualité"""
        print("Initialisation de la génération du dataset professionnel...")
        
        # Calcul des quantités
        num_incident_posts = int(total_size * incident_ratio)
        num_normal_posts = total_size - num_incident_posts
        
        # CORRECTION : Génération de MULTIPLES incidents distincts
        incident_patterns = list(self.incident_patterns.keys())
        incidents_data = []
        incident_counter = 1
        
        # On veut environ 8-12 incidents distincts
        target_num_incidents = random.randint(8, 12)
        posts_per_incident = num_incident_posts // target_num_incidents
        
        for i in range(target_num_incidents):
            pattern = random.choice(incident_patterns)
            # Chaque incident a un nombre variable de posts
            incident_size = random.randint(
                max(5, posts_per_incident - 3), 
                min(25, posts_per_incident + 5)
            )
            
            # Génère l'incident
            new_incident = self.generate_incident(incident_counter, pattern)
            
            # Ajuste la taille si nécessaire
            if len(new_incident) > incident_size:
                new_incident = new_incident[:incident_size]
            
            incidents_data.extend(new_incident)
            incident_counter += 1
        
        # Ajustement si nécessaire (CORRECTEMENT ALIGNÉ)
        incidents_data = incidents_data[:num_incident_posts]
        
        # Génération de l'activité normale
        normal_data = self.generate_normal_activity(num_normal_posts)
        
        # Combinaison des datasets
        all_data = incidents_data + normal_data
        
        # Conversion en DataFrame
        df = pd.DataFrame(all_data)
        
        # Génération des IDs uniques
        df['id'] = [f'REDDIT_POST_{i:06d}' for i in range(1, len(df) + 1)]
        
        # Attribution d'auteurs réalistes
        df['author'] = [random.choice(self.users) for _ in range(len(df))]
        
        # Mapping des subreddits
        df['subreddit'] = df['service'].map(lambda x: self.services[x]['subreddit'])
        
        # Conversion des timestamps
        df['created_utc'] = df['timestamp'].apply(lambda x: x.timestamp())
        df['created_datetime'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Génération du contenu textuel
        print("Génération du contenu textuel...")
        df['text'] = df.apply(self.generate_text_content, axis=1)
        
        # Features supplémentaires
        df = self.calculate_temporal_features(df)
        
        # Réorganisation des colonnes
        column_order = [
            'id', 'title', 'text', 'author', 'subreddit',
            'service', 'component', 'region',
            'created_utc', 'created_datetime', 'timestamp',
            'hour_of_day', 'day_of_week', 'is_weekend', 'days_since_epoch', 'post_density_1h',
            'score', 'num_comments',
            'urgency', 'is_outage',
            'incident_id', 'incident_start', 'incident_pattern', 'post_sequence', 'total_incident_posts'
        ]
        
        df = df[[col for col in column_order if col in df.columns]]
        
        # Mélange final
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def perform_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Exécute des vérifications de qualité sur le dataset"""
        checks = {
            'total_posts': len(df),
            'incident_posts': df['is_outage'].sum(),
            'incident_percentage': (df['is_outage'].sum() / len(df)) * 100,
            'unique_incidents': df['incident_id'].nunique() - 1,  # Exclure NaN
            'services_coverage': df['service'].nunique(),
            'urgency_distribution': df['urgency'].value_counts().to_dict(),
            'temporal_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'missing_values': df.isnull().sum().sum(),
            'text_length_avg': df['text'].str.len().mean()
        }
        
        return checks
    
    def save_dataset(self, df: pd.DataFrame, base_filename: str = 'reddit_cloud_dataset') -> Dict:
        """Sauvegarde le dataset avec documentation"""
        
        # Création du dossier de sortie
        output_dir = 'data/backup'
        os.makedirs(output_dir, exist_ok=True)
        
        # Fichiers à générer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Dataset complet
        full_filename = f"{output_dir}/{base_filename}_full_{timestamp}.csv"
        df.to_csv(full_filename, index=False)
        
        # Dataset d'analyse (colonnes principales)
        analysis_cols = ['id', 'title', 'service', 'component', 'urgency', 'is_outage', 
                        'created_datetime', 'score', 'num_comments', 'incident_id']
        analysis_df = df[analysis_cols]
        analysis_filename = f"{output_dir}/{base_filename}_analysis_{timestamp}.csv"
        analysis_df.to_csv(analysis_filename, index=False)
        
        # Échantillon pour tests rapides
        sample_df = df.sample(min(200, len(df)), random_state=42)
        sample_filename = f"{output_dir}/{base_filename}_sample_{timestamp}.csv"
        sample_df.to_csv(sample_filename, index=False)
        
        # Métadonnées et documentation
        metadata = {
            'generation_date': timestamp,
            'total_records': len(df),
            'files_generated': {
                'full_dataset': full_filename,
                'analysis_dataset': analysis_filename,
                'sample_dataset': sample_filename
            },
            'quality_metrics': self.perform_quality_checks(df),
            'schema': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Sauvegarde des métadonnées
        metadata_filename = f"{output_dir}/{base_filename}_metadata_{timestamp}.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata
    
    def generate_report(self, metadata: Dict):
        """Génère un rapport de génération"""
        
        print("\n" + "="*70)
        print("RAPPORT DE GÉNÉRATION DU DATASET - REDDIT CLOUD MONITOR")
        print("="*70)
        
        print(f"\n📊 STATISTIQUES DU DATASET")
        print("-"*40)
        print(f"Total des posts générés: {metadata['quality_metrics']['total_posts']:,}")
        print(f"Posts d'incident (anomalies): {metadata['quality_metrics']['incident_posts']:,}")
        print(f"Pourcentage d'anomalies: {metadata['quality_metrics']['incident_percentage']:.1f}%")
        print(f"Incidents uniques simulés: {metadata['quality_metrics']['unique_incidents']}")
        
        print(f"\n🏭 COUVERTURE DES SERVICES")
        print("-"*40)
        print(f"Services cloud couverts: {metadata['quality_metrics']['services_coverage']}")
        
        print(f"\n🚨 DISTRIBUTION DES URGENCES")
        print("-"*40)
        for urgency, count in metadata['quality_metrics']['urgency_distribution'].items():
            print(f"  {urgency}: {count:,} posts ({count/metadata['quality_metrics']['total_posts']*100:.1f}%)")
        
        print(f"\n⏱️  CARACTÉRISTIQUES TEMPORELLES")
        print("-"*40)
        print(f"Période couverte: {metadata['quality_metrics']['temporal_range_days']} jours")
        print(f"Longueur moyenne des textes: {metadata['quality_metrics']['text_length_avg']:.0f} caractères")
        
        print(f"\n💾 FICHIERS GÉNÉRÉS")
        print("-"*40)
        for file_type, file_path in metadata['files_generated'].items():
            print(f"  {file_type}: {file_path}")
        
        print(f"\n✅ QUALITÉ DES DONNÉES")
        print("-"*40)
        print(f"Valeurs manquantes totales: {metadata['quality_metrics']['missing_values']}")
        
        print("\n" + "="*70)
        print("GÉNÉRATION TERMINÉE AVEC SUCCÈS")
        print("="*70)

def main():
    """Point d'entrée principal"""
    
    print("Initialisation du générateur de dataset professionnel...")
    
    # Initialisation
    generator = ProfessionalDatasetGenerator()
    
    # Paramètres de génération
    DATASET_SIZE = 800  # Taille totale du dataset
    INCIDENT_RATIO = 0.15  # 15% de posts d'incident
    
    print(f"Paramètres de génération:")
    print(f"  - Taille du dataset: {DATASET_SIZE:,} posts")
    print(f"  - Ratio d'incidents: {INCIDENT_RATIO*100:.0f}%")
    print(f"  - Services couverts: {len(generator.services)}")
    
    # Génération du dataset
    print("\nDébut de la génération du dataset...")
    dataset = generator.create_dataset(
        total_size=DATASET_SIZE,
        incident_ratio=INCIDENT_RATIO
    )
    
    # Sauvegarde
    print("\nSauvegarde des fichiers...")
    metadata = generator.save_dataset(dataset)
    
    # Rapport
    generator.generate_report(metadata)
    
    # Message de succès
    print(f"\n🎯 Le dataset est prêt pour l'analyse Big Data.")
    print(f"📈 Idéal pour: Traitement Spark, Détection d'anomalies, Analyse NLP")
    print(f"🔗 Intégration: Compatible avec PySpark, Pandas, Streamlit")
    
    return dataset

if __name__ == "__main__":
    main()