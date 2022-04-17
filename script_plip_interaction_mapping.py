import json
from json import JSONDecodeError

import requests
from functools import reduce
import pandas as pd
import psutil

from plip.basic import config

from plip.structure.preparation import PDBComplex, PLInteraction
from plip.exchange.report import BindingSiteReport
from plip.basic import config

import dask.dataframe as dd
from pandarallel import pandarallel
from itertools import chain

import logging
import argparse
import sys
import time

from pyspark.sql import SparkSession


def main():

    # establish spark connection
    spark = (
        SparkSession.builder
        .master('local[*]')
        .getOrCreate()
    )

    logging.info('Program begin.')

    config.DNARECEPTOR = True

    logging.info('Loading input.')

    # # Dataset witht all the details, produced earlier:
    # input_dataset = (
    #     pd.read_json(args.input_file, lines=True)
    #     )
    
    input_dataset = (
        spark.read.json(args.input_file)
        .select("pdbStructureId", "chains", "compoundIds")
        .toPandas()
    )

    print(input_dataset)

    pandarallel.initialize(
        nb_workers=psutil.cpu_count(),
        progress_bar=True,
    )

    input_dataset['new_col'] = input_dataset.parallel_apply(
        characerize_complex, axis=1
    )

    logging.info('PLIP interaction computations finished.')

    plip_output_pd_df = pd.DataFrame(list(chain.from_iterable(
        input_dataset
        .loc[lambda df: df.new_col.apply(lambda x: len(x) >0)]
        .assign(new_col = lambda df: df.new_col.apply(lambda l: [value for value in l if value != {}]))
        .new_col
        .to_list()
    )))

    plip_output_pd_df.to_csv(args.output_file, index=False, header=True)

    logging.info('Program finished, file saved as "interaction_structure_drug_plip_output.csv".')


class GetPDB:

    PDB_URL = 'https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{}.ent'

    def __init__(self, data_folder: str) -> None:
        self.data_folder = data_folder


    def get_pdb(self, pdb_structure_id: str) -> str:
        """Reading file from a given loaction fetch and save if not found"""

        logging.info(f'Start searching the file.')

        try:
            # Readind data from the given location:
            with open(f'{self.data_folder}/pdb{pdb_structure_id}.ent', 'rt') as f:
                logging.info(f'try.')
                data = f.read()
                logging.info(f'try ok.')

        except FileNotFoundError:
            logging.info(f'except.')
            # Fetch data from the web
            data = self.fetch_pdb(pdb_structure_id)

            # Save file
            with open(f'{self.data_folder}/pdb{pdb_structure_id}.ent', 'wt') as f:
                f.write(data)
                logging.info(f'except ok.')

        logging.info(f'File obtained.')

        return data


    def fetch_pdb(self, pdb_structure_id: str)-> str:
        """This function fetches the pdb file from ePDB server as a string

        Args:
            pdb_structure_id (str)
        Returns:
            structure data in pdb format as string eg 'AIN:A:1202'
        """
        data = ""

        headers={'Content-Type': 'text/plain'}

        if not pdb_structure_id:
            return ''

        try:
            response = requests.get(self.PDB_URL.format(pdb_structure_id), headers=headers)
            if response.headers['Content-Type'] != 'text/plain; charset=UTF-8':
                pass
            else:
                data = response.text

        except:
            data = ''

        logging.info(f'File obtained or exception while scrapping.')

        return data


def parse_interaction(interaction: PLInteraction, compound_id:str, pdb_id:str) -> dict:

    interaction_type = interaction.__doc__.split('(')[0]

    if interaction_type == 'waterbridge':
        return {}

    # Parsing data form the interaction:
    return {
        'pdb_structure_id': pdb_id,
        'compound_id': compound_id,
        'interaction_type': interaction_type,
        'prot_residue_number': interaction.resnr,
        'prot_residue_type': interaction.restype,
        'prot_chain_id': interaction.reschain
    }



def characerize_complex(row):

    start_time = time.time()

    pdb_id = row[0]
    compounds = row[2]

    logging.info(f'Start characerize_complex: {pdb_id, compounds}')

    gpdb = GetPDB(data_folder=args.pdb_folder)

    pdb_data = gpdb.get_pdb(pdb_id)
    result = []

    if pdb_data:

        # Load into plip:
        mol_complex = PDBComplex()

        try:
            mol_complex.load_pdb(pdb_data, as_string=True)

        except:
            pass

        if mol_complex.ligands:

            # Filtering out only the relevant ligands:
            ligands_of_interest = [ligand for ligand in mol_complex.ligands if ligand.hetid in compounds]

            # Characterizing relevant complex:
            [mol_complex.characterize_complex(ligand) for ligand in ligands_of_interest]

            logging.info(f'Sites computation finished.')

            # Extract details from ligands:
            result = [parse_interaction(interaction, compound.split(':')[0], pdb_id) for compound, interaction_set in mol_complex.interaction_sets.items() for interaction in interaction_set.all_itypes]

    logging.info(f'Done characerize_complex {pdb_id} with ligands {compounds}')
    # Log the complex info data: PDB ID, list of ligands, size of data, processing time in seconds.
    logging.info(f'COMPLEX_INFO\t{pdb_id}\t{compounds}\t{len(pdb_data)}\t{time.time()-start_time}')

    return result


if __name__ == '__main__':

    global spark

    program_description = '''
    Compute PLIP interactions between PDB structures and compounds (Drugs).
    '''

    parser = argparse.ArgumentParser(add_help=True, description=program_description)

    parser.add_argument('-i',
                        '--input_file',
                        help='Path to the csv file with structures and drugs to compute PLIP interactions on.',
                        default=None,
                        metavar='csv_structure_drug_file_path',
                        type=str,
                        required=True)

    parser.add_argument('-o',
                        '--output_file',
                        help='Path to the output csv file with PLIP interactions computed for each structure-drug combinations.',
                        default=None,
                        metavar='csv_plip_output_file_path',
                        type=str,
                        required=True)

    parser.add_argument('-l',
                        '--log_file',
                        help='File to save the logs to.',
                        default=None,
                        metavar='log_file',
                        type=str,
                        required=True)

    parser.add_argument('-f',
                        '--pdb_folder',
                        help='Path to the pdb folder with pdb files downloaded from pdbe website and used by PLIP to compute interactions.',
                        default=None,
                        metavar='pdb_folder_path',
                        type=str,
                        required=True)

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, level=logging.INFO, force=True)

    logging.info(program_description)

    main()
