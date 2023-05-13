#!/bin/bash
User="mininet"
Host="localhost"
SendDir="~/go/src/github.com/mkanakis/middleware/"
Port="2222"

ssh -p $Port $User@$Host "$1"