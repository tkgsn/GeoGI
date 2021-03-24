#!/bin/bash
epsilons=(0.1 0.5 1 3 5 10)
for epsilon in ${epsilons[@]}
do
    python experiments/sql_ae.py --epsilon $epsilon --location Tokyo
done