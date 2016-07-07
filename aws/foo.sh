#!/bin/bash

python heartbeat.py -f common.sh &
HEARTBEAT_PID=$!
sleep 1
kill -TERM $HEARTBEAT_PID
echo sup
