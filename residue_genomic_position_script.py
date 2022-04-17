import argparse
from json import JSONDecodeError
import logging
import psutil

import pandas as pd
import requests
from pandarallel import pandarallel
import pyspark.sql.functions as f
from pyspark.sql import SparkSession


# Global configuration for Spark and Pandarallel.
spark = SparkSession.builder.master('local[*]').getOrCreate()

pandarallel.initialize(
    nb_workers=psutil.cpu_count(),
    progress_bar=True,
)

def main():

    # PLIP INPUT (contain wanted data)
    plip_json_input = (
        # "gene_mapped_structures.json"
        spark.read.json(args.plip_input)

        .select("pdbStructureId", "chains", f.explode("compoundIds").alias("pdbCompoundId"))

        .select("pdbStructureId", "pdbCompoundId", f.explode("chains").alias("chains"))

        )

    plip_json_input_v2 = (plip_json_input

                        .withColumn("chainId", plip_json_input["chains.chainId"])

                        .withColumn("geneId", plip_json_input["chains.geneId"])

                        .withColumn("uniprotId", plip_json_input["chains.uniprot"])

                        .drop("chains")

                        )

    # PLIP OUTPUT
    plip_csv_output = (

        # "output.csv"
        spark.read.csv(args.plip_output, header=True, sep=",")

        .withColumnRenamed("pdb_structure_id", "pdbStructureId")

        .withColumnRenamed("compound_id", "pdbCompoundId")

        .withColumnRenamed("prot_chain_id", "chainId")

        )

    # JOIN to have gene id (used for filter mapping file on the gene id)
    plip_output_target_id = (

        plip_json_input_v2

        .join(plip_csv_output, on=["pdbStructureId", "chainId", "pdbCompoundId"])

        .withColumnRenamed("interaction_type", "intType")

        .withColumnRenamed("prot_residue_number", "protResNb")

        .withColumnRenamed("prot_residue_type", "protResType")

    )

    # Target df
    target_df = (

        spark.read

        .parquet("../targets")

        .select("id", "genomicLocation")

        .withColumn("chromosome", f.col("genomicLocation.chromosome"))
        
        .withColumnRenamed("id", "geneId")

        .drop("genomicLocation")
    )

    plip_output_agg = (

        plip_output_target_id
        
        .join(target_df, on='geneId')

        .groupby([f.col('geneId'),
                f.col('uniprotId'),
                f.col("pdbStructureId").alias("pdbStructId")
                ])

        .agg(f.collect_set(f.col("pdbCompoundId")).alias("pdbCompId"),

            f.collect_set(f.struct(
                f.col('chromosome'),
                f.col('intType'),
                f.col('chainId'),
                f.col('protResType'),
                f.col('protResNb')))        
            .alias("chr, intType, chain, resType, resNb")
            )
        )

    # Test set
    if args.test_set:

        plip_output_agg = plip_output_agg.sample(0.1, 3)

        # plip_output_agg = (
        #     plip_output_agg
        #     .filter(plip_output_agg.pdbStructId.rlike('1dqa'))
        # )

    # Pandas Apply
    genomic_pos_pd = plip_output_agg.toPandas()
    genomic_pos_pd["resNb, resType, chain, intType, chr, genPos"] = genomic_pos_pd.parallel_apply(
        fetch_gapi_ensembl_mapping, axis=1
        )

    # Final DF
    genomic_pos_rm_null_pd = genomic_pos_pd[["geneId", "pdbStructId", "resNb, resType, chain, intType, chr, genPos"]].dropna()

    final_df = genomic_pos_rm_null_pd.explode('resNb, resType, chain, intType, chr, genPos')

    final_df.to_csv(args.output_folder + "/residue_genomic_position.csv", index=False, header=True)


def fetch_gapi_ensembl_mapping(row):
    """This function fetches the graph api ensembl mapping file from ePDB server

    Args:
        rows of the DataFrame (each structures)
    Returns:
        a column for each structure with genomic positions and other infos about residues
    """

    gene_id = row[0]
    pdb_struct_id = row[2]
    residue_info = pd.DataFrame(row[4]).values.tolist()


    url = f'https://www.ebi.ac.uk/pdbe/graph-api/mappings/ensembl/{pdb_struct_id}'
    headers={'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers)

    try:
        if response.headers['Content-Type'] != 'application/json':
            return None

        else:
            e_mapping_file = response.json()
            return filter_dict_file(e_mapping_file, pdb_struct_id, gene_id, residue_info)

    except KeyError:

        return None


def filter_dict_file(e_mapping_file, pdb_struct_id, gene_id, residue_info):

    output_list = []

    e_mapping_dict = e_mapping_file[pdb_struct_id]['Ensembl'][gene_id]['mappings']

    for res in residue_info:

        chromosome = res[0]
        inter_type = res[1]
        chain = res[2]
        res_type = res[3]
        res_nb = int(res[4])

        for res_range in e_mapping_dict:

            start_res = res_range["start"]["author_residue_number"]
            end_res = res_range["end"]["author_residue_number"]

            if start_res <= res_nb and end_res >= res_nb:

                genome_start = res_range["genome_start"]

                res_pos_1 = ((res_nb - start_res) * 3) + genome_start
                res_pos_2 = res_pos_1 + 1
                res_pos_3 = res_pos_1 + 2

                new_elem = [res_nb, res_type, chain, inter_type, chromosome, [res_pos_1, res_pos_2, res_pos_3]]

                if new_elem not in output_list:
                    output_list.append([res_nb, res_type, chain, inter_type, chromosome, [res_pos_1, res_pos_2, res_pos_3]])

    return output_list


if __name__ == '__main__':

    program_description = '''
    Compute genomic position for each residues of each structure of the plip output.
    '''

    parser = argparse.ArgumentParser(add_help=True, description=program_description)

    parser.add_argument('-pi',
                        '--plip_input',
                        help='Path to the csv file with structures and drugs to compute PLIP interactions on.',
                        default=None,
                        metavar='csv_structure_drug_file_path',
                        type=str,
                        required=True)

    parser.add_argument('-po',
                        '--plip_output',
                        help='Path to the output csv file with PLIP interactions computed for each structure-drug combinations.',
                        default=None,
                        metavar='csv_plip_output_file_path',
                        type=str,
                        required=True)

    parser.add_argument('-o',
                        '--output_folder',
                        help='Path to the output folder.',
                        default=None,
                        metavar='path_output_folder',
                        type=str,
                        required=True)

    parser.add_argument('-t',
                        '--test_set',
                        help='Add this argument to run the script on a small set (one structure)',
                        default=None,
                        action='store_true',
                        required=False)

    parser.add_argument('-l',
                        '--log_file',
                        help='File to save the logs to.',
                        default=None,
                        metavar='log_file',
                        type=str,
                        required=False)

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, level=logging.INFO, force=True)

    logging.info(program_description)

    main()
