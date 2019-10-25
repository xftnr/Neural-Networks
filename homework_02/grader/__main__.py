import argparse, importlib

parser = argparse.ArgumentParser('Grade an assignment')
parser.add_argument('assignment', default='homework')
args = parser.parse_args()

print( 'Loading assignment ...' )
assignment = importlib.import_module(args.assignment)

print( 'Loading grader ...' )
from .test_cases import GraderLinear, GraderDeep
g1 = GraderLinear(assignment.linear_model)
g2 = GraderDeep(assignment.deep_model)

print( 'Grading locally ...' )
score1 = g1()
score2 = g2()
print()
print( 'total score locally                                        %3d / %3d'%(score1+score2, g1.TOTAL_SCORE+g2.TOTAL_SCORE) )
