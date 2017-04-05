import logging

from parsing import parsers
from parse_nessus import Nparsers
from single_ip_parsing_nmap import single_ip_parsers

import numpy as np


class Vectorizer:
    """
    This class handles the vectorizing of input
    It additionally stores all pseudo vectors until we are ready for the finished vectors
    """

    def __init__(self):
        self.tokenized_strings = []
        self.pseudo_vectors = {}

    def add_string_to_ip(self, ip, string):
        if ip not in self.pseudo_vectors:
            self.pseudo_vectors[ip] = []

        if string not in self.tokenized_strings:
            self.tokenized_strings.append(string)

        s_id = self.tokenized_strings.index(string)
        self.pseudo_vectors[ip].append(s_id)

    def parse_input(self, input_string, n):

        if (n==True):
            for parser in Nparsers:
                if parser.can_parse_input(input_string):
                    results = parser.parse_input(input_string)
                    for key in results.keys():
                        for s in results[key]:
                            self.add_string_to_ip(key, s)
        else:
            for parser in parsers:
                if parser.can_parse_input(input_string):
                    results = parser.parse_input(input_string)
                    for key in results.keys():
                        for s in results[key]:
                            self.add_string_to_ip(key, s)

    def output_vectors(self):
        vector_names = []
        vectors = np.zeros((len(self.pseudo_vectors.keys()), len(self.tokenized_strings)), dtype=np.float)

        for ip_index, ip in enumerate(self.pseudo_vectors.keys()):
            vector_names.append(ip)
            for s_index in self.pseudo_vectors[ip]:
                # Just set it to one, we want to ignore any case we see a value more than once
                vectors[ip_index, s_index] = 1

        return vector_names, vectors


def vectorize(files_to_vectorize, n):
    vectorizer = Vectorizer()
    for file_path in files_to_vectorize:
        with open(file_path, "r") as f:
            vectorizer.parse_input(f.read(), n)

    vector_names, vectors = vectorizer.output_vectors()

    return vector_names, vectors, vectorizer

def parse_single_ips(files_to_vectorize, ips):
    for file_path in files_to_vectorize:
        with open(file_path, "r") as f:
            for parser in single_ip_parsers:
                #logging.debug("vectorisor selecting single ips")
                results = parser.parse_input(f.read(), ips)
                return results