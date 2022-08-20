from logimage.rule import Rule,RuleSet,RuleList
from logimage.logimage import Logimage
from logimage.input import FileReader
import os

file_path =os.path.abspath("tests/fixtures/rules_example.txt")

rule_set = FileReader(file_path).read_file()
logimage = Logimage(rules=rule_set)

logimage.solve()
logimage.plot_grid()