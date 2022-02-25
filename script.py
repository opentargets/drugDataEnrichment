import requests
import timeit
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
import argparse

spark = SparkSession.builder.getOrCreate()

program_description = '''
This script retrieves the structures in complex with the molecules from parquet.
'''

parser = argparse.ArgumentParser(add_help=True, description=program_description)

parser.add_argument('-p', '--parquet', help='Path to parquet folder', default=None, metavar='parquet_file_path', type=str, required=True)
parser.add_argument('-u', '--unichem', help='Path to unichem src1src3.txt file', default=None, metavar='unichem_file_path', type=str, required=True)
parser.add_argument('-e', '--ensembl', help='Path to pdb_chain_ensembl.csv file', default=None, metavar='ensembl_file_path', type=str, required=True)

args = parser.parse_args()


def get_structure(pdb_id):
    """
    Scrap the list of structures in complex with each molecule from PDBe API
    Function to apply to a df column
    --------------
    Keyword arguments
        pdb_id: id of the molecule for whom we want to know the structures associated
    --------------
    Return
        data[pdb_id] : data is a dictionary, we want the value which is the list of structures
    """
    url = f'https://www.ebi.ac.uk/pdbe/api/pdb/compound/in_pdb/{pdb_id}'
    response = requests.get(url)
    try:
        data = response.json()
        return data[pdb_id]
    except:
        if len(response.json()) == 0:
            return []


def create_pdb_target_gene_df(path_id_file, unichem_molecule_struct_spark_df):

    pdb_chain_ensembl = spark.read.csv(path_id_file, sep=',', header=True, comment='#')
    pdb_chain_ensembl = pdb_chain_ensembl.select('PDB', 'GENE_ID').distinct()

    # print('----- STRUCTURE & TARGET -----')
    # pdb_chain_ensembl.show()

    exploded_df = (unichem_molecule_struct_spark_df
                   .select('MOLECULE_PDB_ID',
                           'MOLECULE_CHEMBL_ID',
                           f.explode(unichem_molecule_struct_spark_df.STRUCTURE_ID)))

    # print('----- STRUCTURE & TARGET EXPLODED -----')
    # exploded_df.show()

    chain_ensembl_struct_mol_joined = (pdb_chain_ensembl
                                       .join(exploded_df, pdb_chain_ensembl["PDB"] == exploded_df["col"])
                                       .drop('col'))

    return chain_ensembl_struct_mol_joined


def main():

    # TIMER 1 START
    start_1 = timeit.default_timer()

    # ----- GET MOLECULE ID -----
    # Molecule DataFrame
    molecule_df = (
        spark.read
        .parquet(args.parquet)
        .select(f.col('id'))
        .withColumnRenamed('id', 'MOLECULE_CHEMBL_ID')
    )

    # ----- GET PDB ID -----
    # Unichem molecules DataFrame
    unichem_df = spark.read.csv(args.unichem, sep=r'\t', header=True)

    # Join Unichem and Molecule DataFrame
    unichem_molecule_df = (molecule_df
                           .join(unichem_df, unichem_df["From src:'1'"] == molecule_df['MOLECULE_CHEMBL_ID'])
                           .withColumnRenamed("To src:'3'", 'MOLECULE_PDB_ID')
                           .drop("From src:'1'")
                           )

    # TIMER 2 START
    start_2 = timeit.default_timer()

    # ----- GET STRUCTURE ID -----
    # Apply get_structure function on the Unichem-Molecule DataFrame

    unichem_molecule_struct_pd_df = (unichem_molecule_df
                                     .toPandas()
                                     .assign(STRUCTURE_ID=lambda x: x['MOLECULE_PDB_ID'].apply(get_structure))
                                     )

    unichem_molecule_struct_pd_df.to_csv('structure_of_molecules.csv', index=False, header=True)

    # Convert Spark DataFrame into Pandas DataFrame
    unichem_molecule_struct_spark_df = spark.createDataFrame(unichem_molecule_struct_pd_df)

    # Count number of structure per molecule
    unichem_molecule_struct_spark_df = (unichem_molecule_struct_spark_df
                                        .withColumn("NB_OF_STRUCT_PER_MOL", f.size(f.col("STRUCTURE_ID")))
                                        .orderBy(f.desc("NB_OF_STRUCT_PER_MOL"))
                                        )

    unichem_molecule_struct_spark_df = (unichem_molecule_struct_spark_df
                                        .filter(f.col('MOLECULE_CHEMBL_ID') != 'CHEMBL692')
                                        .filter(f.col('MOLECULE_CHEMBL_ID') != 'CHEMBL1236970')
                                        .filter(f.col('MOLECULE_CHEMBL_ID') != 'CHEMBL457299')
                                        .filter(f.col('MOLECULE_CHEMBL_ID') != 'CHEMBL113178')
                                        .select('MOLECULE_CHEMBL_ID', 'MOLECULE_PDB_ID', 'STRUCTURE_ID'))

    # Save into a CSV
    unichem_molecule_struct_spark_df.toPandas().to_csv("molecule_structure_target.csv", index=False, header=True)

    # TIMER 2 STOP
    stop_2 = timeit.default_timer()

    # ----- GET TARGET ID -----
    pdb_target_df = create_pdb_target_gene_df(args.ensembl, unichem_molecule_struct_spark_df)

    # ----- STATISTICS -----
    total_nb_unichem = unichem_df.count()
    total_nb_molecule = molecule_df.count()
    nb_mol_in_unichem = unichem_molecule_df.count()

    total_nb_struct = unichem_molecule_struct_spark_df.withColumn('STRUCTURE_ID', f.explode('STRUCTURE_ID')).select(
        'STRUCTURE_ID').distinct().count()

    nb_target_pd = pdb_target_df.count()
    nb_human_target = pdb_target_df.filter(pdb_target_df.GENE_ID.startswith('ENSG')).count()

    count_distinct_target_pd_df = pdb_target_df.groupBy('GENE_ID').count().toPandas()
    nb_molecule_without_struct = unichem_molecule_struct_spark_df.filter(f.size('STRUCTURE_ID') == 0).count()

    nb_non_human_target = nb_target_pd - nb_human_target

    # TIMER 1 STOP
    stop_1 = timeit.default_timer()

    statistics = [
        ['Total molecules in UniChem database', total_nb_unichem],
        ['Molecules in parquet', total_nb_molecule],
        ['Molecules from parquet found in UniChem', nb_mol_in_unichem],
        ['Percentage of parquet molecule found in UniChem', int(round((nb_mol_in_unichem / total_nb_molecule) * 100, 2))],
        ['Total structures', total_nb_struct],
        ['Molecules without structure', nb_molecule_without_struct],
        ['Total targets', nb_target_pd],
        ['Human target', nb_human_target],
        ['None human target', nb_non_human_target],
        ['Total time running', int(round(stop_1 - start_1, 2))],
        ['Time for scrap structure on API', int(round(stop_2 - start_2, 2))]
    ]

    columns = ["Stats", "Count"]

    stats_df = spark.createDataFrame(statistics, columns)

    return stats_df, molecule_df, unichem_df, unichem_molecule_df, unichem_molecule_struct_spark_df, pdb_target_df


if __name__ == '__main__':
    main = main()

    print('----STATISTICS----')
    main[0].show(truncate=False)

    print('----MOLECULE----')
    main[1].show()
    print('----UNICHEM MOLECULE----')
    main[2].show()
    print('----OUR MOLECULE IN UNICHEM (PDB ID)----')
    main[3].show()
    print('----STRUCTURE----')
    main[4].show()
    print('----TARGET----')
    main[5].show()

    print('Process finished!')
