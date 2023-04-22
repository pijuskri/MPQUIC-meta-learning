#!/bin/bash

User="mininet"
Host="localhost"
SourceDir="~/go/src/github.com/lucas-clemente/quic-go/"
SendDir="~/go/src/github.com/lucas-clemente/quic-go/"


scp -P 2222 -r $SourceDir $User@$Host:$SendDir