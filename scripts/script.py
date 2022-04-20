#!/usr/bin/env python
# coding: utf-8

import requests
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
import argparse
from json import JSONDecodeError
import logging
import sys


def main():

    logging.info('Program begin.')

    spark = SparkSession.builder.getOrCreate()

    # ----- GET MOLECULE ID -----
    # Molecule DataFrame
    molecule_df = (
        spark.read
        .parquet(args.parquet)
        .select(
            f.col('inchiKey').alias('inchikey'),
            f.col('id').alias('chemblId'),
            'name', 'linkedTargets', 'linkedDiseases'
        )
    )

    logging.info(f'RESULT: Number of molecules in parquet files (ChemblId): {molecule_df.count()}')

    # ----- GET PDB ID -----
    inchikey_df = (
        spark.read
        .csv(
            args.inchikey,
            sep=',',
            header=True,
            comment='#'
        )
        .select(
            f.col('InChIKey').alias('inchikey'),
            f.col('CCD_ID').alias('pdbCompoundId')
        )
    )

    logging.info(f'RESULT: Number of molecules in Inchikey (inchikeyId): {inchikey_df.count()}')

    # Join Unichem and Molecule DataFrame
    inchikey_molecule_joined_df = (
        molecule_df
        .join(inchikey_df, on='inchikey')
    )

    logging.info('START: Get pdbStructureId on pdbe API.')

    # ----- GET STRUCTURE ID -----
    # Apply get_structure function on the Unichem-Molecule DataFrame
    inchikey_molecule_structure_joined_pd_df = (
        inchikey_molecule_joined_df
        .toPandas()
        .assign(pdbStructureId=lambda df: df.pdbCompoundId.apply(get_structure))
    )

    logging.info('FINISHED: Get pdbStructureId on pdbe API.')

    inchikey_molecule_structure_joined_pd_df.to_csv('output_files/structure_of_molecules.csv', index=False, header=True)

    # Convert Spark DataFrame into Pandas DataFrame
    inchikey_molecule_structure_joined_spark_df = spark.createDataFrame(inchikey_molecule_structure_joined_pd_df)

    # ----- STATISTICS -----
    total_nb_struct_df = (
        inchikey_molecule_structure_joined_spark_df
        .withColumn('pdbStructureId', f.explode('pdbStructureId'))
        .select('pdbStructureId')
        .distinct()
    )

    nb_molecule_without_struct_df = (
        inchikey_molecule_structure_joined_spark_df
        .filter(f.size('pdbStructureId') == 0)
        .distinct()
    )

    logging.info(f'RESULT: Number of structures (pdbId): {total_nb_struct_df.count()}')
    logging.info(f'RESULT: Number of molecule without structure: {nb_molecule_without_struct_df.count()}')

    # Count number of structure per molecule
    inchikey_molecule_structure_joined_spark_df = (
        inchikey_molecule_structure_joined_spark_df
        .withColumn(
            "nbPdbStructure",
            f.size(f.col("pdbStructureId"))
        )
        .orderBy(f.desc("nbPdbStructure"))
    )

    # Save into a CSV
    inchikey_molecule_structure_joined_spark_df.toPandas().to_csv("output_files/molecule_structure_target.csv", index=False, header=True)

    # ----- GET TARGET ID -----
    pdb_target_df = (
            spark.read.csv(args.ensembl, sep=',', header=True, comment='#')
            .select(
                f.col('PDB').alias('pdbStructureId'),
                f.col('GENE_ID').alias('geneId')
            )
            .groupby("pdbStructureId")
            .agg(
                f.collect_set(f.col("geneId")).alias("geneIds")
            )
            .distinct()
            .persist()
    )

    nb_target_pd_df = (
        pdb_target_df
        .withColumn('geneIdsExploded', f.explode('geneIds'))
    )

    nb_human_target_df = (
        pdb_target_df
        .withColumn('geneIdsExploded', f.explode('geneIds'))
        .filter(f.col('geneIdsExploded').startswith('ENSG'))
    )

    logging.info(f'RESULT: Number of targets (ensemblId): {nb_target_pd_df.count()}')
    logging.info(f'RESULT: Number of human targets: {nb_human_target_df.count()}')
    logging.info(f'RESULT: Number of non-human targets: {nb_target_pd_df.count() - nb_human_target_df.count()}')

    inchikey_molecule_structure_target_joined_spark_df = (
        inchikey_molecule_structure_joined_spark_df
        .select("chemblId", "pdbCompoundId", f.explode("pdbStructureId").alias("pdbStructureId"))
        .join(pdb_target_df, on="pdbStructureId", how="inner") # Only keep the structures mapped by ensembl (inner)
    )

    df = (
        inchikey_molecule_structure_target_joined_spark_df
        .groupBy(f.col('chemblId'), f.col('geneIds'))
        .agg(
            f.first(f.col('pdbStructureId')),
            f.first(f.col('pdbCompoundId'))
        )
    )

    df2 = (
        df
        .groupBy(f.col('chemblId'))
        .agg(f.collect_set('first(pdbStructureId)'))
    )

    # Count number of structure per molecule
    df3 = (df2
           .withColumn("nbPdbStructureAfterFilter", f.size(f.col("collect_set(first(pdbStructureId))")))
           .orderBy(f.desc("nbPdbStructureAfterFilter"))
           )

    logging.info('Program finished.')


def get_structure(pdb_compound_id: str) -> list:
    """Fetching structure identifiers from PDBkb REST API

    Args:
        pdb_compound_id: string, a single compound identifier
    Returns:
        List of PDB structure identifiers where the compound can be found
    """
    url = f'https://www.ebi.ac.uk/pdbe/api/pdb/compound/in_pdb/{pdb_compound_id}'
    response = requests.get(url)
    try:
        data = response.json()
        return data[pdb_compound_id]

    except JSONDecodeError:
        print(f'Failed to return structures to: {pdb_compound_id}')
        if len(response.json()) == 0:
            return []

    except KeyError:
        print(f'Empty data was returned for: {pdb_compound_id}')
        return []


def create_pdb_target_gene_df(path_id_file: str, unichem_molecule_struct_df):
    """Updating DataFrame with target identifiers from sift msd as 'chain_ensembl_struct_mol_joined.csv'

    Args:
        path_id_file: string, a path to the csv file
        unichem_molecule_struct_df: dataframe, containing unichem molecule id, pdb compound id, pdbkb structure id
    Returns:
        Same DataFrame than unichem_molecule_struct_spark_df but with the target gene id column in extra
    """
    pdb_chain_ensembl = (
        spark.read.csv(path_id_file, sep=',', header=True, comment='#')
        .select(
            f.col('PDB').alias('pdbStructureId'),
            f.col('CHAIN').alias('chain'),
            f.col('GENE_ID').alias('geneId')
        )
        .distinct()
    )

    return (
        unichem_molecule_struct_df
        # Exploding the STRUCTURE_ID array into a column called PDB:
        .select(
            'chemblId', 'name', 'linkedTargets', 'linkedDiseases', 'pdbCompoundId',
            f.explode(f.col('pdbStructureId')).alias('pdbStructureId')
            )
        # Joining chain ids by PDB identifier:
        .join(pdb_chain_ensembl, on='pdbStructureId', how='inner')
        # The dataframe is stored in memory when returning:
        .persist()
    )


if __name__ == '__main__':

    global spark

    program_description = '''
    This script retrieves the structures corresponding to the complex between molecules from parquet files and targets.
    How to find the files?
    molecule file here (Drug): http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/output/etl/parquet/molecule/*
    inchikey file : https://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/components_inchikeys.csv
    ensembl file : https://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_ensembl.csv
    '''

    parser = argparse.ArgumentParser(add_help=True, description=program_description)

    parser.add_argument('-p',
                        '--parquet',
                        help='Path to parquet folder',
                        default=None,
                        metavar='parquet_file_path',
                        type=str,
                        required=True)

    parser.add_argument('-i',
                        '--inchikey',
                        help='Path to inchikey file',
                        default=None,
                        metavar='inchikey_file_path',
                        type=str,
                        required=True)

    parser.add_argument('-e',
                        '--ensembl',
                        help='Path to pdb_chain_ensembl.csv file',
                        default=None,
                        metavar='ensembl_file_path',
                        type=str,
                        required=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.StreamHandler(sys.stderr)

    logging.info(program_description)

    main()
