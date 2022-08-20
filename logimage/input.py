from logimage.rule import RuleList,RuleSet

class InvalidInputFile(Exception):
    pass

class FileReader:
    def __init__(self, path):
        self.path = path 

    def define_lines(self):
        file1 = open(self.path, 'r')
        self.lines = [line.strip() for line in file1.readlines()]

    def get_row_column_str_indexes(self):
        row_rules_str_index = None
        column_rules_str_index = None
        for index, line in enumerate(self.lines):
            if line == "row_rules":
                row_rules_str_index = index
            if line == "column_rules":
                column_rules_str_index = index
        if (row_rules_str_index == None) or (column_rules_str_index == None):
            raise InvalidInputFile("Invalid File, you must stipulate row_rules and column_rules")
        return row_rules_str_index, column_rules_str_index

    def get_rules_str_lists(self, row_rules_str_index, column_rules_str_index):
        if row_rules_str_index > column_rules_str_index:
            row_rules_str_list = self.lines[row_rules_str_index + 1:]
            column_rules_str_list = self.lines[column_rules_str_index + 1:row_rules_str_index]
        elif row_rules_str_index < column_rules_str_index:
            row_rules_str_list = self.lines[row_rules_str_index + 1:column_rules_str_index]
            column_rules_str_list = self.lines[column_rules_str_index + 1:]
        return row_rules_str_list, column_rules_str_list

    def create_rule_list_from_rules_str_list(self, rules_str_list):
        rule_list = []
        for rule_str in rules_str_list:
            rule_splitted = rule_str.split(",")
            if rule_splitted == [""]:
                rule_splitted_int = []
            else:
                rule_splitted_int = [int(rule_element_str) for rule_element_str in rule_splitted]
            rule_list.append(rule_splitted_int)
        rule_list = RuleList(rule_list)
        return rule_list

    def read_file(self):
        self.define_lines()
        row_rules_str_index, column_rules_str_index = self.get_row_column_str_indexes()
        row_rules_str_list, column_rules_str_list = self.get_rules_str_lists(row_rules_str_index,column_rules_str_index)
        row_rule_list = self.create_rule_list_from_rules_str_list(row_rules_str_list)
        colum_rule_list = self.create_rule_list_from_rules_str_list(column_rules_str_list)
        rule_set = RuleSet(row_rules=row_rule_list,column_rules=colum_rule_list)
        return rule_set
