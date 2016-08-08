#!/bin/bash

WORKDIR="/gpfs/work/rvereeck"
TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M-%S')"`
BACKUPDIR="$WORKDIR/backup"

set -e

# cd $WORKDIR
mkdir -p $BACKUPDIR

tar czvf "$BACKUPDIR/${TIME_STR}.tgz" --exclude='*.pkl' --exclude='*.log' $@
rm -rf $@
