#!/usr/bin/env python3
"""
validate_data.py
Valide la qualité des données générées
"""

import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime
import sys

def validate_dataset(file_path: str) -> dict:
    """Valide un fichier dataset"""
    results = {
        'file': os.path.basename(file_path),
        'valid': False,
        'issues': [],
        'stats': {}
    }
    
    try:
        # Lecture du fichier
        df = pd.read_csv(file_path)
        results['stats']['rows'] = len(df)
        results['stats']['columns'] = len(df.columns)
        
        # Colonnes obligatoires
        required_columns = ['id', 'title', 'service']
        for col in required_columns:
            if col not in df.columns:
                results['issues'].append(f"Missing required column: {col}")
        
        # Vérifie les valeurs nulles
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                results['issues'].append(f"Column {col} has {null_count} null values")
        
        # Vérifie les types de données
        if 'is_outage' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['is_outage']):
                results['issues'].append("Column 'is_outage' should be numeric")
            else:
                # Vérifie les valeurs binaires
                unique_values = df['is_outage'].unique()
                if not all(v in [0, 1] for v in unique_values):
                    results['issues'].append("Column 'is_outage' should only contain 0 or 1")
        
        # Vérifie l'urgence
        if 'urgency' in df.columns:
            valid_urgency = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            invalid_urgency = set(df['urgency'].unique()) - set(valid_urgency)
            if invalid_urgency:
                results['issues'].append(f"Invalid urgency values: {invalid_urgency}")
        
        # Statistiques
        if 'is_outage' in df.columns:
            results['stats']['anomalies'] = int(df['is_outage'].sum())
            results['stats']['anomaly_percentage'] = float((df['is_outage'].sum() / len(df)) * 100)
        
        if 'urgency' in df.columns:
            results['stats']['urgency_distribution'] = df['urgency'].value_counts().to_dict()
        
        if 'service' in df.columns:
            results['stats']['service_distribution'] = df['service'].value_counts().to_dict()
        
        # Détermine si valide
        results['valid'] = len(results['issues']) == 0
        
    except Exception as e:
        results['issues'].append(f"Error reading file: {str(e)}")
    
    return results

def main():
    """Fonction principale"""
    print("🔍 Validation des datasets...")
    
    # Cherche les datasets
    backup_files = glob.glob('data/backup/*.csv')
    live_files = glob.glob('data/live/raw/*.csv') + glob.glob('data/live/processed/*.csv')
    all_files = backup_files + live_files
    
    if not all_files:
        print("❌ Aucun fichier dataset trouvé")
        sys.exit(1)
    
    validation_results = []
    
    for file_path in all_files:
        print(f"\n📁 Validation de: {os.path.basename(file_path)}")
        result = validate_dataset(file_path)
        validation_results.append(result)
        
        if result['valid']:
            print(f"   ✅ VALIDE: {result['stats']['rows']} rows, {result['stats']['columns']} cols")
            if 'anomalies' in result['stats']:
                print(f"   🚨 Anomalies: {result['stats']['anomalies']} ({result['stats']['anomaly_percentage']:.1f}%)")
        else:
            print(f"   ❌ INVALIDE: {len(result['issues'])} issues")
            for issue in result['issues'][:3]:  # Affiche seulement 3 premiers problèmes
                print(f"      • {issue}")
    
    # Rapport final
    valid_count = sum(1 for r in validation_results if r['valid'])
    total_count = len(validation_results)
    
    print(f"\n{'='*60}")
    print("📊 RAPPORT DE VALIDATION")
    print("="*60)
    print(f"Fichiers analysés: {total_count}")
    print(f"Fichiers valides: {valid_count}")
    print(f"Taux de succès: {(valid_count/total_count*100):.1f}%")
    
    if valid_count == total_count:
        print("\n🎉 TOUS LES DATASETS SONT VALIDES !")
        sys.exit(0)
    else:
        print("\n❌ Certains datasets ont des problèmes")
        sys.exit(1)

if __name__ == "__main__":
    main()