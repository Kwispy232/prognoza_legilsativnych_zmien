#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kontrola závislostí pre SARIMA analýzu
"""

def check_dependencies():
    """Kontrola dostupnosti knižníc potrebných pre SARIMA analýzu"""
    missing_libs = []
    
    # Zoznam potrebných knižníc
    required_libs = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'scipy',
        'sklearn'
    ]
    
    # Kontrola knižníc
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✓ Knižnica {lib} je k dispozícii")
        except ImportError:
            print(f"✗ Knižnica {lib} chýba")
            missing_libs.append(lib)
    
    # Výsledok
    if missing_libs:
        print("\nPre SARIMA analýzu chýbajú nasledujúce knižnice:")
        for lib in missing_libs:
            print(f"- {lib}")
        print("\nPre inštaláciu závislostí môžete použiť príkaz:")
        print(f"pip install {' '.join(missing_libs)}")
        return False
    else:
        print("\nVšetky potrebné knižnice sú k dispozícii.")
        print("Môžete spustiť SARIMA analýzu pomocou príkazu:")
        print("python legislation/sarima_complete.py")
        return True

if __name__ == "__main__":
    print("Kontrola závislostí pre SARIMA analýzu...\n")
    check_dependencies()
