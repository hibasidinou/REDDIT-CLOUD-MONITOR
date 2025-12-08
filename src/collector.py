"""
collector.py
Collecteur professionnel de données Reddit Cloud Monitor
Version complète avec classe, analyse NLP et fonctionnalités avancées
"""

import pandas as pd
import numpy as np
import yaml
import os
import glob
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RedditCollector:
    """
    Collecteur professionnel pour l'analyse de pannes cloud
    
    DIFFÉRENCES AVEC LA VERSION SIMPLE :
    ✅ Classe orientée objet (meilleure organisation)
    ✅ Analyse NLP intégrée (sentiment, urgence, services)
    ✅ Gestion des erreurs robuste
    ✅ Configuration externe (config.yaml)
    ✅ Simulation temps réel intelligente
    ✅ Métriques et statistiques détaillées
    ✅ Sauvegarde organisée (raw/processed)
    ✅ Logging et monitoring
    ✅ Extensible pour API Reddit future
    """
    
    def __init__(self, config_path: str = 'config.yaml', use_backup: bool = True):
        """
        Initialise le collecteur
        
        Args:
            config_path: Chemin vers le fichier de configuration
            use_backup: Si True, utilise les données backup au lieu d'API
        """
        self.config = self._load_config(config_path)
        self.use_backup = use_backup
        self.dataset_cache = None
        self.stats = {}
        
        print(f"🔧 RedditCollector initialisé")
        print(f"   Mode: {'Backup' if use_backup else 'API'}")
        print(f"   Config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis un fichier YAML"""
        default_config = {
            'data': {
                'backup_dir': 'data/backup',
                'output_dir': 'data/live',
                'cache_enabled': True
            },
            'analysis': {
                'urgency_levels': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                'cloud_services': ['AWS', 'Azure', 'GitHub', 'Google Cloud'],
                'keywords': {
                    'outage': ['down', 'outage', 'broken', 'crash', 'failed'],
                    'degradation': ['slow', 'lag', 'issue', 'problem'],
                    'normal': ['discussion', 'question', 'help', 'tutorial']
                }
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                # Fusionne avec la config par défaut
                config = self._deep_merge(default_config, user_config)
                print(f"✅ Configuration chargée: {config_path}")
            else:
                config = default_config
                print(f"⚠️  Utilisation config par défaut")
        except Exception as e:
            print(f"❌ Erreur chargement config: {e}")
            config = default_config
        
        return config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Fusionne récursivement deux dictionnaires"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def load_latest_dataset(self) -> pd.DataFrame:
        """
        Charge le dataset backup le plus récent
        Retourne un DataFrame pandas avec toutes les colonnes
        """
        backup_dir = self.config['data']['backup_dir']
        
        # Cherche les fichiers de dataset
        patterns = [
            'reddit_cloud_dataset_analysis_*.csv',
            'reddit_cloud_dataset_full_*.csv',
            'reddit_professional.csv',
            'reddit_sample.csv'
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            search_path = os.path.join(backup_dir, pattern)
            files = glob.glob(search_path)
            
            for file in files:
                file_time = os.path.getctime(file)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file
        
        if not latest_file:
            raise FileNotFoundError(f"Aucun dataset trouvé dans {backup_dir}")
        
        try:
            df = pd.read_csv(latest_file)
            print(f"📂 Dataset chargé: {os.path.basename(latest_file)}")
            print(f"📊 Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
            
            # Cache le dataset pour réutilisation
            if self.config['data'].get('cache_enabled', True):
                self.dataset_cache = df
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur lecture {latest_file}: {e}")
            raise
    
    def analyze_urgency_nlp(self, text: str) -> Tuple[str, float]:
        """
        Analyse l'urgence d'un texte avec NLP simple
        
        Args:
            text: Texte à analyser
            
        Returns:
            Tuple (niveau_urgence, score_confiance)
        """
        if not isinstance(text, str):
            return 'LOW', 0.0
        
        text_lower = text.lower()
        scores = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        # Mots-clés pondérés
        keywords = {
            'CRITICAL': [
                ('down', 3.0), ('outage', 3.0), ('critical', 2.5), ('emergency', 2.5),
                ('broken', 2.0), ('crash', 2.0), ('failed', 2.0), ('disaster', 2.0)
            ],
            'HIGH': [
                ('severe', 1.5), ('major', 1.5), ('urgent', 1.5), ('unavailable', 1.5),
                ('not working', 1.5), ('help', 1.0), ('issue', 1.0)
            ],
            'MEDIUM': [
                ('slow', 1.0), ('lag', 1.0), ('problem', 1.0), ('degraded', 1.0),
                ('intermittent', 1.0), ('partial', 1.0)
            ],
            'LOW': [
                ('discussion', 0.5), ('question', 0.5), ('tutorial', 0.5),
                ('best practice', 0.5), ('learning', 0.5)
            ]
        }
        
        # Calcul des scores
        for level, level_keywords in keywords.items():
            for keyword, weight in level_keywords:
                if keyword in text_lower:
                    scores[level] += weight
        
        # Détection de ponctuation d'urgence
        if '!' in text:
            scores['CRITICAL'] += 1.0
        if '?' in text:
            scores['MEDIUM'] += 0.5
        
        # Détection de majuscules (URGENT)
        if len(text) > 10:
            upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if upper_ratio > 0.3:  # Plus de 30% de majuscules
                scores['CRITICAL'] += 2.0
        
        # Détermination du niveau d'urgence
        max_score = max(scores.values())
        if max_score == 0:
            return 'LOW', 0.0
        
        # Trouve le niveau avec le score max
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if scores[level] == max_score:
                confidence = min(max_score / 5.0, 1.0)  # Normalise entre 0 et 1
                return level, confidence
        
        return 'LOW', 0.0
    
    def extract_cloud_service(self, text: str) -> str:
        """Extrait le service cloud mentionné dans le texte"""
        if not isinstance(text, str):
            return 'Unknown'
        
        text_lower = text.lower()
        
        services_mapping = {
            'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'rds', 'cloudfront'],
            'Azure': ['azure', 'microsoft azure', 'azure devops', 'blob storage', 'sql azure'],
            'GitHub': ['github', 'git hub', 'github actions', 'github pages'],
            'Google Cloud': ['google cloud', 'gcp', 'google cloud platform', 'bigquery', 'gke'],
            'DigitalOcean': ['digitalocean', 'droplet', 'spaces'],
            'Cloudflare': ['cloudflare']
        }
        
        for service, keywords in services_mapping.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return service
        
        return 'Unknown'
    
    def enrich_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit le dataset avec des analyses NLP
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame enrichi avec nouvelles colonnes
        """
        print("🔍 Enrichissement du dataset avec NLP...")
        
        # Copie pour éviter les modifications sur l'original
        df_enriched = df.copy()
        
        # Analyse d'urgence NLP
        if 'title' in df_enriched.columns:
            urgency_results = df_enriched['title'].apply(self.analyze_urgency_nlp)
            df_enriched['nlp_urgency'] = urgency_results.apply(lambda x: x[0])
            df_enriched['nlp_confidence'] = urgency_results.apply(lambda x: x[1])
        
        # Extraction de service
        if 'title' in df_enriched.columns:
            df_enriched['detected_service'] = df_enriched['title'].apply(self.extract_cloud_service)
        
        # Longueur du texte
        if 'title' in df_enriched.columns:
            df_enriched['title_length'] = df_enriched['title'].apply(len)
        
        # Heure de la journée
        if 'created_datetime' in df_enriched.columns:
            try:
                df_enriched['post_hour'] = pd.to_datetime(df_enriched['created_datetime']).dt.hour
                df_enriched['is_business_hours'] = df_enriched['post_hour'].between(9, 17).astype(int)
            except:
                pass
        
        print(f"✅ Dataset enrichi: {len(df_enriched.columns)} colonnes")
        return df_enriched
    
    def simulate_realtime_stream(self, df: pd.DataFrame, hours: int = 72) -> pd.DataFrame:
        """
        Simule un flux temps réel à partir du dataset historique
        
        Args:
            df: Dataset complet
            hours: Nombre d'heures à simuler en 'temps réel' (72h = 3 jours)
            
        Returns:
            Sous-ensemble simulant les posts récents
        """
        print(f"⏱️  Simulation flux temps réel ({hours}h = {hours/24:.1f} jours)...")
        
        try:
            # Identifie la colonne de temps
            time_col = None
            for col in ['created_datetime', 'timestamp', 'created_utc']:
                if col in df.columns:
                    time_col = col
                    break
            
            if not time_col:
                print("⚠️  Aucune colonne de temps trouvée, retour de 100 posts aléatoires")
                return df.sample(min(100, len(df)), random_state=42)
            
            # Convertit en datetime
            df_temp = df.copy()
            df_temp['_temp_datetime'] = pd.to_datetime(df_temp[time_col])
            
            # Calcule la date la plus récente
            latest_time = df_temp['_temp_datetime'].max()
            cutoff_time = latest_time - timedelta(hours=hours)
            
            # Filtre les posts récents
            recent_posts = df_temp[df_temp['_temp_datetime'] > cutoff_time].copy()
            
            if len(recent_posts) > 0:
                # Supprime la colonne temporaire
                recent_posts = recent_posts.drop('_temp_datetime', axis=1)
                print(f"📡 {len(recent_posts)} posts récents simulés (dernières {hours}h)")
                
                # Si moins de 50 posts, ajoute des posts aléatoires
                if len(recent_posts) < 50:
                    print(f"⚠️  Peu de posts récents, ajout de posts aléatoires...")
                    # Prend des posts anciens aléatoires
                    old_posts = df_temp[df_temp['_temp_datetime'] <= cutoff_time].copy()
                    if not old_posts.empty:
                        num_to_add = min(100 - len(recent_posts), len(old_posts))
                        additional_posts = old_posts.sample(num_to_add, random_state=42)
                        additional_posts = additional_posts.drop('_temp_datetime', axis=1)
                        
                        # Combine
                        combined = pd.concat([recent_posts, additional_posts], ignore_index=True)
                        print(f"📦 Total: {len(combined)} posts (récents + aléatoires)")
                        return combined
                
                return recent_posts
            else:
                print("ℹ️  Aucun post récent, retour de 150 posts aléatoires")
                return df.sample(min(150, len(df)), random_state=42)
                
        except Exception as e:
            print(f"⚠️  Erreur simulation: {e}")
            return df.sample(min(150, len(df)), random_state=42)
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcule des statistiques détaillées sur le dataset"""
        print("📈 Calcul des statistiques...")
        
        stats = {
            'total_posts': len(df),
            'columns': list(df.columns),
            'data_types': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Statistiques par colonne si disponible
        if 'is_outage' in df.columns:
            stats['outage_posts'] = int(df['is_outage'].sum())
            stats['outage_percentage'] = float((df['is_outage'].sum() / len(df)) * 100)
        
        if 'urgency' in df.columns:
            urgency_counts = df['urgency'].value_counts().to_dict()
            stats['urgency_distribution'] = urgency_counts
        
        if 'service' in df.columns:
            service_counts = df['service'].value_counts().to_dict()
            stats['service_distribution'] = service_counts
        
        if 'score' in df.columns:
            stats['avg_score'] = float(df['score'].mean())
            stats['max_score'] = int(df['score'].max())
        
        # Enrichissements NLP si disponibles
        if 'nlp_urgency' in df.columns:
            nlp_urgency_counts = df['nlp_urgency'].value_counts().to_dict()
            stats['nlp_urgency_distribution'] = nlp_urgency_counts
        
        if 'nlp_confidence' in df.columns:
            stats['avg_nlp_confidence'] = float(df['nlp_confidence'].mean())
        
        self.stats = stats
        return stats
    
    def save_to_output(self, df: pd.DataFrame, prefix: str = 'collected'):
        """
        Sauvegarde les données dans le dossier de sortie
        
        Args:
            df: DataFrame à sauvegarder
            prefix: Préfixe pour le nom du fichier
        """
        output_dir = self.config['data']['output_dir']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Crée les dossiers
        raw_dir = os.path.join(output_dir, 'raw')
        processed_dir = os.path.join(output_dir, 'processed')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Sauvegarde raw (toutes les colonnes)
        raw_file = os.path.join(raw_dir, f'{prefix}_raw_{timestamp}.csv')
        df.to_csv(raw_file, index=False)
        
        # Sauvegarde processed (colonnes essentielles)
        essential_cols = ['id', 'title', 'service', 'urgency', 'is_outage', 'created_datetime']
        available_cols = [col for col in essential_cols if col in df.columns]
        
        if available_cols:
            processed_df = df[available_cols]
            processed_file = os.path.join(processed_dir, f'{prefix}_processed_{timestamp}.csv')
            processed_df.to_csv(processed_file, index=False)
            print(f"💾 Données traitées: {processed_file}")
        
        # Sauvegarde des statistiques
        stats_file = os.path.join(output_dir, f'{prefix}_stats_{timestamp}.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"💾 Données brutes: {raw_file}")
        print(f"💾 Statistiques: {stats_file}")
    
    def collect(self, simulate_realtime: bool = True, enrich: bool = True) -> pd.DataFrame:
        """
        Méthode principale pour collecter et traiter les données
        
        Args:
            simulate_realtime: Si True, simule un flux temps réel
            enrich: Si True, enrichit avec NLP
            
        Returns:
            DataFrame traité
        """
        print("\n" + "="*60)
        print("🚀 COLLECTE DE DONNÉES - REDDIT CLOUD MONITOR")
        print("="*60)
        
        try:
            # 1. Chargement du dataset
            df = self.load_latest_dataset()
            
            # 2. Simulation temps réel
            if simulate_realtime:
                df = self.simulate_realtime_stream(df)
            
            # 3. Enrichissement NLP
            if enrich:
                df = self.enrich_dataset(df)
            
            # 4. Calcul des statistiques
            stats = self.calculate_statistics(df)
            
            # 5. Sauvegarde
            self.save_to_output(df)
            
            # 6. Rapport
            self._print_report(stats)
            
            print("\n✅ Collecte terminée avec succès!")
            return df
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la collecte: {e}")
            raise
    
    def _print_report(self, stats: Dict):
        """Affiche un rapport détaillé"""
        print("\n" + "="*60)
        print("📊 RAPPORT DE COLLECTE")
        print("="*60)
        
        print(f"📈 POSTS ANALYSÉS: {stats.get('total_posts', 0):,}")
        
        if 'outage_posts' in stats:
            print(f"🚨 ANOMALIES DÉTECTÉES: {stats['outage_posts']:,} ({stats['outage_percentage']:.1f}%)")
        
        if 'urgency_distribution' in stats:
            print("\n🎯 DISTRIBUTION URGENCE:")
            for level, count in stats['urgency_distribution'].items():
                percentage = (count / stats['total_posts']) * 100
                print(f"  {level}: {count:,} posts ({percentage:.1f}%)")
        
        if 'service_distribution' in stats:
            print("\n🏭 SERVICES CLOUD:")
            for service, count in stats['service_distribution'].items():
                print(f"  {service}: {count:,} posts")
        
        if 'nlp_urgency_distribution' in stats:
            print("\n🤖 ANALYSE NLP:")
            for level, count in stats['nlp_urgency_distribution'].items():
                print(f"  {level}: {count:,} posts")
        
        print("="*60)

# Fonction de commodité pour une utilisation simple
def collect_data(simulate_realtime: bool = True) -> pd.DataFrame:
    """
    Fonction simplifiée pour collecter des données
    
    Args:
        simulate_realtime: Si True, simule un flux temps réel
        
    Returns:
        DataFrame avec les données collectées
    """
    collector = RedditCollector()
    return collector.collect(simulate_realtime=simulate_realtime)

def main():
    """Point d'entrée principal"""
    print("🧪 TEST DU COLLECTEUR PROFESSIONNEL")
    print("="*60)
    
    try:
        # Initialise le collecteur
        collector = RedditCollector()
        
        # Collecte les données
        df = collector.collect(simulate_realtime=True, enrich=True)
        
        # Affiche un aperçu
        print("\n📋 APERÇU DES DONNÉES:")
        if not df.empty:
            preview_cols = ['title', 'service', 'urgency', 'is_outage']
            available_cols = [col for col in preview_cols if col in df.columns]
            
            if available_cols:
                print(df[available_cols].head().to_string())
            
            print(f"\n🎯 Le collecteur est prêt pour l'intégration avec Spark!")
            print(f"   Total posts disponibles: {len(df)}")
            print(f"   Colonnes: {len(df.columns)}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("💡 Vérifiez que le dossier data/backup contient des datasets")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)