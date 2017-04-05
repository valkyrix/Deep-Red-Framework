import logging
import xml.etree.ElementTree as ET
from cStringIO import StringIO
from json import dumps


class NessusXMLParser:
    def __init__(self):
        pass

    def can_parse_input(self, input_string):

        return input_string.startswith("<?xml") and "NessusClientData" in input_string

    def parse_input(self, input_string):
        """
        Parse the Nessus XML input and dump out the list of strings representing out vectors
        :param input_string:
        :return:
        """
        logging.info("Parsing Nessus XML * BETA *")

        tree = ET.parse(StringIO(input_string))
        root = tree.getroot()

        ip_addresses = {}

        vulnerabilities = dict()
        for block in root:
            if block.tag == "Report":
                # each host in report
                for report_host in block:
                    host_properties_dict = dict()
                    for report_item in report_host:
                        if report_item.tag == "HostProperties":
                            for host_properties in report_item:
                                host_properties_dict[host_properties.attrib['name']] = host_properties.text

                                # add hosts who dont have vulnerabilities
                                ip_address = report_host.attrib['name']
                                if ip_address not in ip_addresses:
                                    ip_addresses[ip_address] = []

                    for report_item in report_host:
                        if 'pluginName' in report_item.attrib:

                            vulner_id = report_host.attrib['name'] + "|" + report_item.attrib['pluginID'] + "|" + \
                                        report_item.attrib['port']
                            vulnerabilities[vulner_id] = dict()

                            vulnerabilities[vulner_id]['port'] = report_item.attrib['port']
                            vulnerabilities[vulner_id]['pluginName'] = report_item.attrib['pluginName']
                            vulnerabilities[vulner_id]['pluginFamily'] = report_item.attrib['pluginFamily']
                            vulnerabilities[vulner_id]['pluginID'] = report_item.attrib['pluginID']

                            # port=report_item.attrib['port']
                            # pluginName=report_item.attrib['pluginName']
                            # pluginFamily=report_item.attrib['pluginFamily']
                            # pluginID=report_item.attrib['pluginID']
                            #
                            # s = dumps([port, pluginName, pluginFamily, pluginID])
                            # ip_addresses[ip].append(s)

                            for param in report_item:
                                if param.tag == "risk_factor":
                                    risk_factor = param.text

                                    vulnerabilities[vulner_id]['host'] = report_host.attrib['name']
                                    vulnerabilities[vulner_id]['riskFactor'] = risk_factor

                                    # s = dumps([report_host.attrib['name'], risk_factor])
                                    # ip_addresses[ip].append(s)

                                else:
                                    vulnerabilities[vulner_id][param.tag] = param.text

                                    # s = dumps(param.tag, param.text)
                                    # ip_addresses[ip].append(s)

                            for param in host_properties_dict:
                                # takes all host properties to be appended to vulnerability features
                                vulnerabilities[vulner_id][param] = host_properties_dict[param]

                                if (host_properties_dict[param] == "Advanced Scan") or (
                                    host_properties_dict[param] == "false"):
                                    continue
                                else:
                                    # prints out all used host parameters
                                    #print param
                                    s = dumps(host_properties_dict[param])
                                    ip_addresses[vulnerabilities[vulner_id]["host"]].append(s)

        for vulner_id in vulnerabilities:

            # adds ip addresses into a array which will become our json array to cluster
            ip_address = vulnerabilities[vulner_id]["host"]
            if ip_address not in ip_addresses:
                ip_addresses[ip_address] = []

            # convert vulns to json. only those with a risk value and cvss vector
            if "riskFactor" in vulnerabilities[vulner_id].keys() and "cvss_vector" in vulnerabilities[
                vulner_id].keys():
                # prints used vulnerability data
                # print(vulner_id + " - " + vulnerabilities[vulner_id]["plugin_name"] + " - Risk_Factor: " +
                #       vulnerabilities[vulner_id]['riskFactor'])

                # uses dumps, a json obj converter.
                # this is what will be used to calculate labels, features, and run through PCA before being clustered
                # s = dumps(vulnerabilities[vulner_id]['port'], vulnerabilities[vulner_id]['cvss_vector'], vulnerabilities[vulner_id]['riskFactor'], vulnerabilities[vulner_id]['pluginID'], vulnerabilities[vulner_id]['pluginFamily'], vulnerabilities[vulner_id]["plugin_name"])
                s = dumps(vulnerabilities[vulner_id]['port'], vulnerabilities[vulner_id]['cvss_base_score'],
                          vulnerabilities[vulner_id]['riskFactor'], vulnerabilities[vulner_id]['plugin_name'],
                          vulnerabilities[vulner_id]['pluginFamily'])

                ip_addresses[ip_address].append(s)

                # following if statement only selects critical vulnerabilities
                #
                # elif "riskFactor" in vulnerabilities[vulner_id].keys() and "cvss_vector" in vulnerabilities[
                #     vulner_id].keys() and "exploit_available" in vulnerabilities[vulner_id].keys():
                #     print("vulnerabilities found with exploit:")
                #     print(vulner_id + " " + vulnerabilities[vulner_id]["plugin_name"])
                #     if vulnerabilities[vulner_id]["riskFactor"] == "Critical" and "AV:N" in vulnerabilities[vulner_id][
                #         "cvss_vector"] and vulnerabilities[vulner_id]["exploit_available"] == "true":
                #         print(vulner_id + " " + vulnerabilities[vulner_id]["plugin_name"])

        print "no of IP's taken from nessus: " + str(len(ip_addresses.viewkeys()))
        logging.info("Done Nessus parsing")
        return ip_addresses


Nparsers = [
    NessusXMLParser()
]
