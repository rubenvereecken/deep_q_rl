export STATE=$1
export NAME=$2

qstat -n1 | tail -n +6 | tr -s ' ' | awk -F' ' -v NAME=$NAME -v STATE=$STATE 'match($4, NAME) && match($10, STATE) {print $1}'
#qstat -n1 | tail -n +6 | tr -s ' ' | cut -d' ' -f1 
