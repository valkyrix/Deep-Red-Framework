@echo off
title Execute nessus scan from premade policy

ipython "%~dp0\nessus-scan.py" -- --target 192.168.0.4 --policy testPolicy --name 'demo' --insecure
type "%~dp0\scanTest.txt"
ipython "%~dp0\nessus_report_exporter.py" -- < "%~dp0\scanTest.txt"

echo End
pause