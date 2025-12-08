#!/usr/bin/env python3
"""
run_pipeline.py
Exécute le pipeline de données complet
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd: str, description: str) -> bool:
    """Exécute une commande et vérifie le succès"""
    print(f"\n▶️  {description}")
    print(f"   Commande: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Succès")
            if result.stdout:
                print(f"   Sortie: {result.stdout[:200]}...")
            return True
        else:
            print(f"   ❌ Échec (code: {result.returncode})")
            print(f"   Erreur: {result.stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main(test_mode: bool = False):
    """Exécute le pipeline complet"""
    print("🚀 LANCEMENT DU PIPELINE DE DONNÉES")
    print("="*60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/pipeline_{timestamp}.log"
    
    # Crée le dossier logs
    os.makedirs('logs', exist_ok=True)
    
    steps = [
        # Étape 1: Génération des données
        {
            'cmd': 'python src/generate_backup.py',
            'desc': 'Génération du dataset'
        },
        
        # Étape 2: Collecte des données
        {
            'cmd': 'python src/collector.py',
            'desc': 'Collecte et enrichissement'
        },
        
        # Étape 3: Validation
        {
            'cmd': 'python scripts/validate_data.py',
            'desc': 'Validation des données'
        },
        
        # Étape 4: Test Spark (si installé)
        {
            'cmd': 'python -c "from src.processor import DataProcessor; print(\'✅ Spark compatible\')"',
            'desc': 'Test compatibilité Spark'
        },
        
        # Étape 5: Génération du dashboard
        {
            'cmd': 'python src/dashboard.py --generate-report',
            'desc': 'Génération du rapport'
        }
    ]
    
    if test_mode:
        # Mode test - étapes réduites
        steps = steps[:3]
        print("🔧 MODE TEST ACTIVÉ")
    
    # Exécute chaque étape
    all_success = True
    
    for i, step in enumerate(steps, 1):
        print(f"\n📋 Étape {i}/{len(steps)}")
        success = run_command(step['cmd'], step['desc'])
        
        if not success:
            all_success = False
            if not test_mode:
                print(f"\n❌ Pipeline arrêté à l\'étape {i}")
                break
    
    # Rapport final
    print(f"\n{'='*60}")
    print("📊 RAPPORT DU PIPELINE")
    print("="*60)
    
    if all_success:
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
        print(f"\n📁 Fichiers générés:")
        print(f"   • data/backup/ - Datasets de sauvegarde")
        print(f"   • data/live/ - Données collectées")
        print(f"   • logs/ - Journaux d'exécution")
        print(f"\n🎯 Prêt pour l'analyse Big Data !")
    else:
        print("❌ PIPELINE ÉCHOUÉ")
        print("💡 Vérifiez les logs pour les détails")
    
    print("="*60)
    return all_success

if __name__ == "__main__":
    test_mode = '--test' in sys.argv
    success = main(test_mode)
    sys.exit(0 if success else 1)