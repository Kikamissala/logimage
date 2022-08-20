from logimage.input import FileReader, InvalidInputFile
import os
import pytest

from logimage.rule import RuleList, Rule, RuleSet

def test_file_reader_raises_error_when_file_not_compatible():
    file_path = os.path.abspath("tests/fixtures/bad_rules_example.txt")
    file_reader = FileReader(path = file_path)
    with pytest.raises(InvalidInputFile) as err:
        file_reader.read_file()

def test_file_reader_get_row_column_rules_index():
    file_path = os.path.abspath("tests/fixtures/rules_example.txt")
    file_reader = FileReader(path = file_path)
    file_reader.define_lines()
    row_rules_str_index, column_rules_str_index = file_reader.get_row_column_str_indexes()
    assert((row_rules_str_index == 0) & (column_rules_str_index == 3))

def test_file_reader_get_row_column_rules_index_when_column_before():
    file_path = os.path.abspath("tests/fixtures/rules_example_inverted.txt")
    file_reader = FileReader(path = file_path)
    file_reader.define_lines()
    row_rules_str_index, column_rules_str_index = file_reader.get_row_column_str_indexes()
    assert((column_rules_str_index == 0) & (row_rules_str_index == 3))

def test_file_reader_gets_rule_str_lists():
    file_path = os.path.abspath("tests/fixtures/rules_example.txt")
    file_reader = FileReader(path = file_path)
    file_reader.define_lines()
    row_rules_str_index = 0
    column_rules_str_index = 3
    row_rules_str_list, column_rules_str_list = file_reader.get_rules_str_lists(row_rules_str_index,column_rules_str_index)
    expected_row_rules_str_list = ["2","2"]
    expected_column_rules_str_list = ["2","2"]
    assert((row_rules_str_list == expected_row_rules_str_list) & (column_rules_str_list == expected_column_rules_str_list))

def test_file_reader_gets_rule_lists():
    file_path = os.path.abspath("tests/fixtures/rules_example.txt")
    file_reader = FileReader(path = file_path)
    row_rules_str_list = ["1,1","2"]
    rule_list = file_reader.create_rule_list_from_rules_str_list(rules_str_list=row_rules_str_list)
    expected_rule_list = RuleList([Rule([1,1]),Rule([2])])
    assert(expected_rule_list == rule_list)

def test_file_reader_gets_rule_lists():
    file_path = os.path.abspath("tests/fixtures/rules_example.txt")
    file_reader = FileReader(path = file_path)
    rule_set = file_reader.read_file()
    expected_rule_set = RuleSet(row_rules = RuleList([Rule([2]),Rule([2])]),column_rules=RuleList([Rule([2]),Rule([2])]))
    assert(rule_set == expected_rule_set)