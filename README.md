# Logimage / Picross Solving

This repo is a custom implementation of a picross solver in Python.

It is still a work in progress. The solver solves any picross but for now the performance is not on point.

## Install

To install the package, first install poetry. (Link [here](https://python-poetry.org/))

Then clone the repo, and do, in the directory :

```
poetry install
poetry shell
```

## picross file

To represent a grid to solve with clues, we use a txt file like this : 

```
row_rules
2,2
column_rules
2,2
```

Some examples in [rules_examples](/rules_examples)

## Launch a solve

```
python solve_logimage.md /path/to/picross.txt
```
