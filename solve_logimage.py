from logimage.logimage import Logimage
from logimage.input import FileReader
import argparse
import os

def solve_logimage(file_path):
    file_path =os.path.abspath(file_path)

    rule_set = FileReader(file_path).read_file()
    logimage = Logimage(rules=rule_set,guessing_heuristic="least_undefined")
    logimage.solve()
    logimage.plot_grid()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rule_file', type=str, help='Input rules txt path')
    args = parser.parse_args()

    solve_logimage(args.rule_file)