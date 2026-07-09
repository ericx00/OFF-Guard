"""
Quick manual test for sanctions_check.py.

Run with:
    python test_sanctions.py

First run will download and cache the real OFAC SDN list (a few MB),
so it needs an internet connection once. After that it's cached locally
under data/sdn_cache.csv for a day.
"""

from sanctions_check import check_name

TEST_NAMES = [
    "Hamza Bin Laden",       # confirmed present in your data as "BIN LADEN, Hamza"
    "Sa'ad Bin Laden",       # confirmed present as "BIN LADEN, Sa'ad"
    "AeroCaribbean Airlines",  # confirmed present, an entity not a person
    "Random Person XYZ123",  # should have no match
]

if __name__ == "__main__":
    for name in TEST_NAMES:
        print(f"\nChecking: {name}")
        matches = check_name(name)
        if not matches:
            print("  No matches found.")
        else:
            for m in matches:
                print(f"  MATCH: {m.name}  (score={m.score:.0f}, program={m.program}, id={m.entity_number})")