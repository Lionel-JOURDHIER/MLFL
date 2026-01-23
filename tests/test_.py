#
#! mettre le nom du fichier à tester après le "test_"
#! par exemple pour tester main.py appeler le fichier test_main.py

import pytest
#! importer les focntions à tester dans le fichier. 
#! exemple : from app.modules.zorglangue import zorg_trad


@pytest.mark.parametrize(
    
    "input, expected", 

    [
    # Exemples donnés
    ("Bonjour", "ruojnoB"),
    ("Vive Zorglub !", "eviV bulgroZ !"),
    ("Ceci est un message secret", "iceC tse nu egassem terces"),

    ]
)

def test_examples(inp, expected):
#! mettre les fonction à tester 
#!  assert zorg_trad(inp) == expected
#!  asssert True à supprimer    
    assert True