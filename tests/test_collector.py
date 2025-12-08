"""
test_collector.py
Tests unitaires pour le module collector.py
"""

import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import yaml

# Ajoute le répertoire src au path Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from collector import RedditCollector

class TestRedditCollector(unittest.TestCase):
    """Tests pour la classe RedditCollector"""
    
    def setUp(self):
        """Setup avant chaque test"""
        # Crée un dossier temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.backup_dir = os.path.join(self.test_dir, 'data', 'backup')
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Crée un dataset de test minimal
        self.create_test_dataset()
        
        # Crée un fichier config.yaml de test
        self.create_test_config()
        
        # Initialise le collecteur avec le dossier de test
        os.environ['TEST_MODE'] = 'True'
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if 'TEST_MODE' in os.environ:
            del os.environ['TEST_MODE']
    
    def create_test_dataset(self):
        """Crée un dataset de test pour les tests unitaires"""
        # Dataset minimal de 50 posts
        data = []
        services = ['AWS', 'Azure', 'GitHub']
        
        for i in range(50):
            is_outage = i % 10 == 0  # 10% d'anomalies (5 posts)
            service = services[i % 3]
            
            data.append({
                'id': f'test_post_{i:03d}',
                'title': f"{'URGENT: ' if is_outage else ''}{service} service {'outage' if is_outage else 'discussion'}",
                'service': service,
                'urgency': 'HIGH' if is_outage else 'LOW',
                'is_outage': 1 if is_outage else 0,
                'created_datetime': (datetime.now() - timedelta(hours=i)).isoformat(),
                'score': 100 if is_outage else 20,
                'num_comments': 50 if is_outage else 5,
                'incident_id': f'INC-{i//10:03d}' if is_outage else None,
                'source': 'test_dataset'
            })
        
        df = pd.DataFrame(data)
        
        # Sauvegarde le dataset de test
        test_file = os.path.join(self.backup_dir, 'test_dataset.csv')
        df.to_csv(test_file, index=False)
        
        # Crée aussi un fichier avec le pattern attendu
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_file = os.path.join(self.backup_dir, f'reddit_cloud_dataset_analysis_{timestamp}.csv')
        df.to_csv(analysis_file, index=False)
        
        self.test_dataset_path = test_file
        self.analysis_dataset_path = analysis_file
    
    def create_test_config(self):
        """Crée un fichier config.yaml de test"""
        config_content = """
app:
  subreddits: ["aws", "azure", "github"]
  keywords: ["down", "outage", "error", "slow"]
  check_interval: 60
  history_days: 7

data:
  backup_dir: "data/backup"
  output_dir: "data/live"

reddit:
  client_id: "test_client_id"
  client_secret: "test_client_secret"
  user_agent: "TestAgent/1.0"
"""
        
        config_path = os.path.join(self.test_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Modifie le collecteur pour utiliser ce config
        original_load_config = RedditCollector.load_config
        
        def mock_load_config(self):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Modifie le backup_dir pour pointer vers notre dossier de test
            config['data']['backup_dir'] = self.backup_dir
            return config
        
        self.original_load_config = original_load_config
        RedditCollector.load_config = mock_load_config
    
    def test_01_collector_initialization(self):
        """Test l'initialisation du collecteur"""
        print("\n🧪 Test 1: Initialisation du collecteur")
        
        # Test sans API
        collector = RedditCollector(use_api=False)
        self.assertIsNotNone(collector)
        self.assertFalse(collector.use_api)
        self.assertIsNone(collector.reddit_client)
        
        # Test avec API (mais pas de vrai client)
        collector_with_api = RedditCollector(use_api=True)
        self.assertIsNotNone(collector_with_api)
        self.assertTrue(collector_with_api.use_api)
        
        print("✅ Initialisation réussie")
    
    def test_02_config_loading(self):
        """Test le chargement de la configuration"""
        print("\n🧪 Test 2: Chargement de la configuration")
        
        collector = RedditCollector(use_api=False)
        
        # Vérifie que la config est chargée
        self.assertIsInstance(collector.config, dict)
        self.assertIn('app', collector.config)
        self.assertIn('data', collector.config)
        
        # Vérifie les valeurs spécifiques
        self.assertEqual(len(collector.config['app']['subreddits']), 3)
        self.assertIn('aws', collector.config['app']['subreddits'])
        
        print(f"✅ Configuration chargée: {len(collector.config['app']['subreddits'])} subreddits")
    
    def test_03_load_backup_dataset(self):
        """Test le chargement du dataset backup"""
        print("\n🧪 Test 3: Chargement du dataset backup")
        
        collector = RedditCollector(use_api=False)
        df = collector.load_latest_backup_dataset()
        
        # Vérifie que le DataFrame n'est pas vide
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Vérifie le nombre de posts
        self.assertEqual(len(df), 50)
        
        # Vérifie les colonnes essentielles
        required_columns = ['id', 'title', 'service', 'urgency', 'is_outage']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Vérifie les statistiques
        anomalies = df['is_outage'].sum()
        self.assertEqual(anomalies, 5)  # 10% de 50 = 5 anomalies
        
        print(f"✅ Dataset chargé: {len(df)} posts, {anomalies} anomalies")
    
    def test_04_urgency_analysis(self):
        """Test l'analyse d'urgence"""
        print("\n🧪 Test 4: Analyse d'urgence")
        
        collector = RedditCollector(use_api=False)
        
        # Test avec différents textes
        test_cases = [
            ("CRITICAL: AWS EC2 down in us-east-1", "CRITICAL"),
            ("Major outage on Azure services", "HIGH"),
            ("Performance issues with GitHub Actions", "MEDIUM"),
            ("Discussion about cloud best practices", "LOW"),
            ("Help needed with AWS configuration", "LOW")
        ]
        
        for text, expected_urgency in test_cases:
            urgency = collector.analyze_urgency(text)
            self.assertEqual(urgency, expected_urgency)
        
        print("✅ Analyse d'urgence fonctionnelle")
    
    def test_05_service_extraction(self):
        """Test l'extraction de service cloud"""
        print("\n🧪 Test 5: Extraction de service cloud")
        
        collector = RedditCollector(use_api=False)
        
        test_cases = [
            ("AWS EC2 instance failing", "AWS"),
            ("Azure Blob Storage issues", "Azure"),
            ("GitHub Actions not working", "GitHub"),
            ("Google Cloud Platform outage", "Google Cloud"),
            ("DigitalOcean droplet problems", "DigitalOcean"),
            ("General discussion about cloud", "Unknown")
        ]
        
        for text, expected_service in test_cases:
            service = collector.extract_cloud_service(text)
            self.assertEqual(service, expected_service)
        
        print("✅ Extraction de service fonctionnelle")
    
    def test_06_outage_detection(self):
        """Test la détection de pannes"""
        print("\n🧪 Test 6: Détection de pannes")
        
        collector = RedditCollector(use_api=False)
        
        test_cases = [
            ("AWS is down right now", 1),
            ("Major outage reported", 1),
            ("Service not working", 1),
            ("Performance discussion", 0),
            ("Best practices guide", 0),
            ("How to optimize costs", 0)
        ]
        
        for text, expected_is_outage in test_cases:
            is_outage = collector.is_outage_post(text)
            self.assertEqual(is_outage, expected_is_outage)
        
        print("✅ Détection de pannes fonctionnelle")
    
    def test_07_simulate_real_time(self):
        """Test la simulation temps réel"""
        print("\n🧪 Test 7: Simulation temps réel")
        
        collector = RedditCollector(use_api=False)
        df = collector.load_latest_backup_dataset()
        
        # Test la simulation
        realtime_df = collector.simulate_real_time_collection(df, hours_back=24)
        
        # Vérifie que le résultat n'est pas vide
        self.assertFalse(realtime_df.empty)
        
        # Vérifie que c'est un sous-ensemble du dataset original
        self.assertLessEqual(len(realtime_df), len(df))
        
        print(f"✅ Simulation temps réel: {len(realtime_df)} posts récents")
    
    def test_08_collect_data_method(self):
        """Test la méthode principale collect_data"""
        print("\n🧪 Test 8: Méthode collect_data")
        
        collector = RedditCollector(use_api=False)
        
        # Test en mode backup
        df = collector.collect_data(mode='backup', simulate_realtime=False)
        
        # Vérifications
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 50)
        self.assertIn('source', df.columns)
        
        # Vérifie les colonnes présentes
        expected_cols = ['id', 'title', 'service', 'urgency', 'is_outage']
        for col in expected_cols:
            self.assertIn(col, df.columns)
        
        print(f"✅ collect_data réussie: {len(df)} posts collectés")
    
    def test_09_save_collected_data(self):
        """Test la sauvegarde des données collectées"""
        print("\n🧪 Test 9: Sauvegarde des données")
        
        collector = RedditCollector(use_api=False)
        df = collector.collect_data(mode='backup', simulate_realtime=False)
        
        # Sauvegarde les données
        output_dir = os.path.join(self.test_dir, 'data', 'live')
        collector.config['data']['output_dir'] = output_dir
        
        collector.save_collected_data(df, output_type='both')
        
        # Vérifie que les fichiers ont été créés
        raw_files = [f for f in os.listdir(os.path.join(output_dir, 'raw')) if f.endswith('.csv')]
        processed_files = [f for f in os.listdir(os.path.join(output_dir, 'processed')) if f.endswith('.csv')]
        
        self.assertGreater(len(raw_files), 0)
        self.assertGreater(len(processed_files), 0)
        
        # Vérifie le contenu des fichiers
        raw_file_path = os.path.join(output_dir, 'raw', raw_files[0])
        raw_df = pd.read_csv(raw_file_path)
        self.assertEqual(len(raw_df), len(df))
        
        print(f"✅ Données sauvegardées: {len(raw_files)} fichier(s) raw, {len(processed_files)} fichier(s) processed")
    
    def test_10_minimal_dataset_fallback(self):
        """Test le fallback sur dataset minimal"""
        print("\n🧪 Test 10: Fallback dataset minimal")
        
        # Supprime les fichiers de dataset pour forcer le fallback
        for f in os.listdir(self.backup_dir):
            os.remove(os.path.join(self.backup_dir, f))
        
        collector = RedditCollector(use_api=False)
        
        # Cette méthode devrait créer un dataset minimal
        df = collector.load_latest_backup_dataset()
        
        # Vérifications
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 0)
        self.assertIn('source', df.columns)
        self.assertTrue(all(df['source'] == 'minimal_backup'))
        
        print(f"✅ Fallback réussi: {len(df)} posts créés")
    
    def test_11_statistics_generation(self):
        """Test la génération de statistiques"""
        print("\n🧪 Test 11: Génération de statistiques")
        
        collector = RedditCollector(use_api=False)
        df = collector.collect_data(mode='backup', simulate_realtime=False)
        
        # Vérifie que les statistiques de base sont correctes
        total_posts = len(df)
        anomalies = df['is_outage'].sum() if 'is_outage' in df.columns else 0
        
        self.assertEqual(total_posts, 50)
        self.assertEqual(anomalies, 5)  # 10% de 50
        
        # Vérifie la distribution des services
        if 'service' in df.columns:
            service_counts = df['service'].value_counts()
            self.assertIn('AWS', service_counts.index)
            self.assertIn('Azure', service_counts.index)
            self.assertIn('GitHub', service_counts.index)
        
        print(f"✅ Statistiques: {total_posts} posts, {anomalies} anomalies")
    
    def test_12_integration_with_pandas(self):
        """Test l'intégration avec pandas"""
        print("\n🧪 Test 12: Intégration pandas")
        
        # Charge le dataset directement avec pandas
        df = pd.read_csv(self.analysis_dataset_path)
        
        # Vérifie les types de données
        self.assertIsInstance(df, pd.DataFrame)
        
        # Opérations pandas basiques
        filtered = df[df['is_outage'] == 1]
        grouped = df.groupby('service')['score'].mean()
        
        self.assertEqual(len(filtered), 5)  # 5 anomalies
        self.assertGreater(len(grouped), 0)
        
        print("✅ Intégration pandas réussie")
    
    def test_13_data_quality_checks(self):
        """Test les vérifications de qualité des données"""
        print("\n🧪 Test 13: Qualité des données")
        
        df = pd.read_csv(self.analysis_dataset_path)
        
        # Vérifie l'absence de valeurs nulles dans les colonnes critiques
        critical_columns = ['id', 'title', 'service', 'is_outage']
        
        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                self.assertEqual(null_count, 0, f"Colonne {col} a {null_count} valeurs nulles")
        
        # Vérifie les types de données
        if 'is_outage' in df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(df['is_outage']))
        
        if 'score' in df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(df['score']))
        
        print("✅ Qualité des données validée")

def run_all_tests():
    """Exécute tous les tests et génère un rapport"""
    print("="*70)
    print("🧪 SUITE DE TESTS - REDDIT COLLECTOR")
    print("="*70)
    
    # Crée un test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRedditCollector)
    
    # Exécute les tests avec un runner personnalisé
    runner = unittest.TextTestRunner(verbosity=2, descriptions=True)
    
    print("\n▶️  Démarrage des tests...\n")
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "="*70)
    print("📊 RAPPORT DES TESTS")
    print("="*70)
    
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    
    print(f"Total tests: {total}")
    print(f"✅ Tests réussis: {passed}")
    print(f"❌ Tests échoués: {len(result.failures)}")
    print(f"⚠️  Tests avec erreurs: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 TOUS LES TESTS ONT RÉUSSI !")
        print("Le collecteur est prêt pour la production.")
    else:
        print("\n🔧 Certains tests nécessitent des corrections.")
    
    print("="*70)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Exécute tous les tests
    success = run_all_tests()
    
    # Code de sortie pour CI/CD
    sys.exit(0 if success else 1)