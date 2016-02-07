#!/bin/bash

env | sort > ~/test.txt
env | sort > /shared/`hostname`-test.txt
