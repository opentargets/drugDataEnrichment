#!/usr/bin/env python
# coding: utf-8

# # This script retrieves the structures in complex with the molecules from parquet.How to find the files?
# 
# #### molecule file (Drug) : [http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/output/etl/parquet/molecule/*]()
# #### unichem file : [https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src3.txt.gz]()
# #### ensembl file : [https://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_ensembl.csv]()

# In[41]:


import requests
import timeit
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
import argparse


# In[42]:


program_description = '''
This script retrieves the structures in complex with the molecules from parquet.
How to find the files?
molecule file here (Drug): http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/output/etl/parquet/molecule/*
unichem file : https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src3.txt.gz
ensembl file : https://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_ensembl.csv
'''

parser = argparse.ArgumentParser(add_help=True, description=program_description)

parser.add_argument('-p', '--parquet', help='Path to parquet folder', default=None, metavar='parquet_file_path', type=str, required=True)
parser.add_argument('-u', '--unichem', help='Path to unichem src1src3.txt file', default=None, metavar='unichem_file_path', type=str, required=True)
parser.add_argument('-e', '--ensembl', help='Path to pdb_chain_ensembl.csv file', default=None, metavar='ensembl_file_path', type=str, required=True)

args = parser.parse_args()


# In[43]:


molecule_path = '/Users/marinegirardey/Documents/OpenTargetInternship/molecule/'
# molecule = '/Users/marinegirardey/Documents/OpenTargetInternship/small_folder/'
unichem_path = '/Users/marinegirardey/Documents/OpenTargetInternship/id_files/src1src3.txt'
ensembl_path = '/Users/marinegirardey/Documents/OpenTargetInternship/chain_ensembl_struct_mol_joined.csv'

dropped_molecules = [
    'CHEMBL692', 'CHEMBL1236970', 'CHEMBL457299', 'CHEMBL113178'
]


# In[ ]:


spark = SparkSession.builder.getOrCreate()


# In[ ]:


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
    except:
        if len(response.json()) == 0:
            return []


# In[ ]:


def create_pdb_target_gene_df(path_id_file: str, unichem_molecule_struct_spark_df):
    """Updating DataFrame with target identifiers from sift msd as 'chain_ensembl_struct_mol_joined.csv'

    Args:
        path_id_file: string, a path to the csv file
        unichem_molecule_struct_spark_df: dataframe, containing unichem molecule id, pdb compound id, pdbkb structure id
    Returns:
        Same DataFrame than unichem_molecule_struct_spark_df but with the target gene id column in extra
    """
    pdb_chain_ensembl = (
        spark.read.csv(path_id_file, sep=',', header=True, comment='#')
            .select('PDB', 'GENE_ID').distinct()
    )

    return (
        unichem_molecule_struct_spark_df

            # Exploding the STRUCTURE_ID array into a column called PDB:
            .select(
                'name', 'CHEMBL_MOLECULE_ID', 'PDB_COMPOUND_ID',
                f.explode(f.col('STRUCTURE_ID')).alias('PDB')
            )

            # Joining chain ids by PDB identifier:
            .join(pdb_chain_ensembl, on='PDB', how='inner')

            # The dataframe is stored in memory when returning:
            .persist()
    )


# In[ ]:


# TIMER 1 START
start_1 = timeit.default_timer()


# In[ ]:


# ----- GET MOLECULE ID -----
# Molecule DataFrame
molecule_df = (
    spark.read
    .parquet(args.parquet)
    .select(f.col('name').alias('NAME'), f.col('id').alias('CHEMBL_MOLECULE_ID'))
)
molecule_df.show()


# In[ ]:


# ----- GET PDB ID -----
# Unichem molecules DataFrame
unichem_df = (
    spark.read.csv(args.unichem, sep=r'\t', header=True)
    .withColumnRenamed('From src:\'1\'', 'CHEMBL_MOLECULE_ID')
    .withColumnRenamed('To src:\'3\'', 'PDB_COMPOUND_ID')
)

# Join Unichem and Molecule DataFrame
unichem_molecule_df = (molecule_df.alias("a")
                       .join(unichem_df.alias("b"), unichem_df["CHEMBL_MOLECULE_ID"] == molecule_df['CHEMBL_MOLECULE_ID'])
                       ).select("NAME", "a.CHEMBL_MOLECULE_ID", "PDB_COMPOUND_ID")

unichem_molecule_df.show()


# In[ ]:


# TIMER 2 START
start_2 = timeit.default_timer()


# In[ ]:


# ----- GET STRUCTURE ID -----
# Apply get_structure function on the Unichem-Molecule DataFrame

unichem_molecule_struct_pd_df = (unichem_molecule_df
                                 .toPandas()
                                 .assign(STRUCTURE_ID=lambda x: x['PDB_COMPOUND_ID'].apply(get_structure))
                                 )


# 

# In[ ]:


unichem_molecule_struct_pd_df.to_csv('structure_of_molecules.csv', index=False, header=True)


# In[ ]:


# Convert Spark DataFrame into Pandas DataFrame
unichem_molecule_struct_spark_df = spark.createDataFrame(unichem_molecule_struct_pd_df)
unichem_molecule_struct_spark_df.show()


# In[ ]:


# Count number of structure per molecule
unichem_molecule_struct_spark_df = (unichem_molecule_struct_spark_df
                                    .withColumn("NB_OF_STRUCT_PER_MOL", f.size(f.col("STRUCTURE_ID")))
                                    .orderBy(f.desc("NB_OF_STRUCT_PER_MOL"))
                                    )
unichem_molecule_struct_spark_df.show()


# In[ ]:


unichem_molecule_struct_spark_df = (
    unichem_molecule_struct_spark_df
        .filter(~f.col('CHEMBL_MOLECULE_ID').isin(dropped_molecules))
        .select('*')
)
unichem_molecule_struct_spark_df.show()


# In[ ]:


# Save into a CSV
unichem_molecule_struct_spark_df.toPandas().to_csv("molecule_structure_target.csv", index=False, header=True)


# In[ ]:


# TIMER 2 STOP
stop_2 = timeit.default_timer()


# In[ ]:


# ----- GET TARGET ID -----
pdb_target_df = create_pdb_target_gene_df(args.ensembl, unichem_molecule_struct_spark_df)
pdb_target_df.show()


# In[ ]:


# ----- STATISTICS -----
total_nb_unichem = unichem_df.count()
total_nb_molecule = molecule_df.count()
nb_mol_in_unichem = unichem_molecule_df.count()


# In[ ]:


total_nb_struct = unichem_molecule_struct_spark_df.withColumn('STRUCTURE_ID', f.explode('STRUCTURE_ID')).select(
    'STRUCTURE_ID').distinct().count()


# In[ ]:


nb_target_pd = pdb_target_df.count()
nb_human_target = pdb_target_df.filter(pdb_target_df.GENE_ID.startswith('ENSG')).count()


# In[ ]:


nb_molecule_without_struct = unichem_molecule_struct_spark_df.filter(f.size('STRUCTURE_ID') == 0).count()


# In[ ]:


nb_non_human_target = nb_target_pd - nb_human_target


# In[ ]:


# TIMER 1 STOP
stop_1 = timeit.default_timer()


# In[ ]:


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


# In[ ]:


stats_df = spark.createDataFrame(statistics, columns)


# In[ ]:


print('----STATISTICS----')
stats_df.show(truncate=False)

# print('----MOLECULE----')
# molecule_df.show()

# print('----UNICHEM MOLECULE----')
# unichem_df.show()

# print('----OUR MOLECULE IN UNICHEM (PDB ID)----')
# unichem_molecule_df.show()

# print('----STRUCTURE----')
# unichem_molecule_struct_spark_df.show()

# print('----TARGET----')
# pdb_target_df.show()

