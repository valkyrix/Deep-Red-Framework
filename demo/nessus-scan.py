#!/usr/bin/env python

# Example implementation for scanning a specific target with an already existing policy.

import ness6rest
import argparse

# Settings
nessus_url = "https://127.0.0.1:8834"
login = "script";
password = "script";

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target',  required=True)
parser.add_argument('--policy', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--insecure', action="store_true")
parser.add_argument('--ca_bundle')
args = parser.parse_args()

# Log in
scan = ness6rest.Scanner(url=nessus_url, login=login, password=password, insecure=args.insecure, ca_bundle=args.ca_bundle)

# Set policy that should be used
scan.policy_set(name=args.policy)

# Set target and scan name
scan.scan_add(targets=args.target, name=args.name)

# Run scan

print "Starting scan now, UUID will follow:"
scan.scan_run_red()
##prints uuid to file so exporter can read it
with open('scanTest.txt', 'w') as file_:
    file_.write(scan.scan_uuid)
scan._scan_status()

