from logimage.rule import RuleSet, InvalidRule
from logimage.logimage import Logimage
import pytest

def test_generate_logimage_needs_grid_dimensions_and_list_of_rules_for_rows_and_columns():
    logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_a_rule_values_exceeds_size_of_grid():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1,1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_number_of_row_rules_not_equal_number_of_rows():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1],[1],[1]],column_rules = [[1],[1]]))

# créer un moyen de générer l'ensemble des problèmes. object avec les problèmes row et colonnes
# il faut coder la dépendance entre un problème ligne et l'ensemble des pb colonnes et inversement.
# si on modifie un problème, la modification doit être automatiquement reportée sur les colonnes
# à savoir qu'on ne peut pas modifier un problème nimporte comment.
# la modification d'un problème ne peut pas créer une erreur dans les problèmes dépendants car les 
# seules modifications dûes à du solve remplacent du undefined par du defined
# Il faut juste une fonction d'update qui utilise un setitem dans les problèmes et va update
# Il faut aussi une fonction de check de cohérence, qui en gros est une double boucle qui vérifie que
# RowProblems[i][j] == ColumnProblems[j][i] pour i nb lignes et j nombre de colonnes
# la fonction qui va répercuter de l'un à l'autre ou inversement devra être la même
# et une répercussion ressemblera à : RowProblems[i][j] = ColumnProblems[j][i] avec i ou j fixé
# Si c'est une ligne modifée, i constant et j parcourant toutes les colonnes (ou juste celles modifiées)
# pour le solve en lui même, on peut utiliser l'une ou l'autre des fonctions, peu importe


# pour identifier quelles lignes / colonnes solve : 
# Dans un premier temps on regarde si c'est solvable direct => on solve les pb qui peuvent l'être et on 
# impacte les autres.
# Puis on a une deuxième vague avec les overlap potentiels
# Si on a aucun problème fully solvable, ni overlap => pas solvable sans guess
# On crée les 2 listes, et si elles sont vides on met un raise
# si un élément est dans liste de fully defined, il sort celle de l'overlap (on gère ça avec if elif else)


# il faut une liste des pb à solve, mise à jour régulièrement.
# Cette liste contient au départ toutes les lignes et colonnes impactées par les deux premières loops
# à chaque nouveau round de solve, on parcourt cette liste et on regarde pour chacun de ses éléments
# quels sont les problèmes avec le moins de cases undefined.
# tous les problèmes sans cases undefined sont supprimés de la liste car solved
# puis le premier avec le plus faible nombre de cases undefined est choisi pour le solve.
# Quand il est solve, il est exclu de la liste jusqu'à ce qu'il soit potentiellement modifié à nouveau.


# Il faut un moyen de gérer les éléments de cette liste, et pouvoir associer une valeur de cette liste avec
# un problème via des coordonnées. (clés avec 2 valeurs la première si ligne ou colonne, la deuxième l'index)
# le set semble parfait.


# Chaque début de run, on déduit les clés des problèmes impactés par la résolution du problème précédent, et
# on contrôle si un de ces problèmes est maintenant solved. Dans ce cas on le sort du set.
# Puis sur les autres problèmes on regarde le nombre de degrés de liberté. On peut aussi tenir à jour cette valeur
# dans un dict avec la clé étant les coordonnées, la valeur le nombre de degrés de liberté
# on prend la valeur minimum de degrés de liberté et on solve ce problème.
# quand le problème est solved, on analyse le nombre de cases modifiées, on retient leur index et on fait deux choses:
# on impacte les modifs sur les problèmes liés grâce aux index
# si le problème impacté n'est pas déjà dans la liste, on le rajoute
# s'il est déjà dedans, on update son nombre de cases libres, et si cette valeur est 0, on sort le pb de la liste





def test_create_initial_problems_():
    pass