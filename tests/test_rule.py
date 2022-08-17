from logimage.cell import Cell, CellState
from logimage.rule import RuleList,RuleElement, Rule, RuleSet

def test_translate_rule_element_of_element_1_returns_list_of_len_1_with_1():
    rule_element = RuleElement(1)
    assert(rule_element.translate_to_list() == [Cell(CellState.full)])

def test_translate_rule_element_of_element_2_returns_list_of_len_2_with_1():
    rule_element = RuleElement(2)
    assert(rule_element.translate_to_list() == [Cell(CellState.full),Cell(CellState.full)])

def test_check_if_elements_of_rule_are_rule_elements():
    rule = Rule([1,1])
    list_of_bool = [isinstance(element,RuleElement) for element in rule]
    assert(all(list_of_bool) == True)

def test_compute_rule_minimum_possible_line_len():
    rule = Rule([1,1])
    minimum_possible_line_len = rule.compute_min_possible_len()
    assert(minimum_possible_line_len == 3)

def test_compute_rule_minimum_possible_line_len_for_unique_rule_element():
    rule = Rule([2])
    minimum_possible_line_len = rule.compute_min_possible_len()
    assert(minimum_possible_line_len == 2)

def test_check_if_elements_of_rulelist_are_rules():
    rule_list = RuleList([[1,1],[1,1]])
    list_of_bool = [isinstance(element,Rule) for element in rule_list]
    assert(all(list_of_bool) == True)

def test_compute_maximum_minimum_possible_len_of_rule_list():
    rule_list = RuleList([[1,1],[1,1,1]])
    maximum_minimum_possible_len = rule_list.compute_maximum_minimum_possible_len()
    assert(maximum_minimum_possible_len == 5)

def test_rules_creation_with_row_rules_and_column_rules():
    row_rules = [[1],[1]]
    column_rules = [[1],[1]]
    rules = RuleSet(row_rules = row_rules, column_rules = column_rules)
    assert((rules.row_rules == row_rules) & (rules.column_rules == column_rules))