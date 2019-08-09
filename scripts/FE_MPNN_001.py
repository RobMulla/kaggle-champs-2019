import os
from rdkit import Chem
import openbabel as ob
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
obConversion = ob.OBConversion()
_ = obConversion.SetInAndOutFormats("xyz", "mol2")

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ss = StandardScaler()
train['scc_scaled'] = ss.fit_transform(train[['scalar_coupling_constant']])
mean_scc = train.groupby('molecule_name')['scc_scaled'].mean()
max_size = 29

def read_xyz(mol, index_of_mu = 4):
    file_path = f'/home/robmulla/Repos/gated-graph-neural-network-samples/data/qm9_raw/{mol}.xyz'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        smiles = lines[-2].split('\t')[0]
        properties = lines[1].split('\t')
        mu = float(properties[index_of_mu])
        #average_scc = mean_scc[mol]
        ob_mol = ob.OBMol()
        obConversion.ReadFile(ob_mol, file_path)
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
    # return {'smiles': smiles, 'mu': mu, 'average_scc': average_scc, 'ob_mol': ob_mol, 'rdkit_mol': rdkit_mol}
    return {'smiles': smiles, 'mu': mu, 'ob_mol': ob_mol, 'rdkit_mol': rdkit_mol}

count = 0
with open('angles_train.csv', mode='a') as file_:
    # Write the CSV header
    file_.write('molecule_name,atom_index_0,atom_index_1,ang0,ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,ang10,ang11,ang12,ang13,ang14,ang15,ang16,ang17,ang18,ang19,ang20,ang21,ang22,ang23,ang24,ang25,ang26,ang27,ang28')
    file_.write("\n")

for mol_name in tqdm(train['molecule_name'].unique()):
    dat = read_xyz(mol_name)
    for atom1 in ob.OBMolAtomIter(dat['ob_mol']):
        for atom0 in ob.OBMolAtomIter(dat['ob_mol']):
            anglist = []
            for atom2 in ob.OBMolAtomIter(dat['ob_mol']):
                ang = atom0.GetAngle(atom1, atom2)
                ang /= 180
                anglist.append(ang)
            atom_0 = atom0.GetIdx() - 1
            atom_1 = atom1.GetIdx() - 1
            anglist.extend([0] * (max_size - len(anglist))) # Extend ang list
            anglist_commas = str(anglist)[1:-1]
            csv_line = f'{mol_name},{atom_0},{atom_1},{anglist_commas}'
            with open('angles_train.csv', mode='a') as file_:
                file_.write(csv_line)
                file_.write("\n")
#     if count == 100:
#         break
#     count += 1
with open('angles_test.csv', mode='a') as file_:
    # Write the CSV header
    file_.write('molecule_name,atom_index_0,atom_index_1,ang0,ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,ang10,ang11,ang12,ang13,ang14,ang15,ang16,ang17,ang18,ang19,ang20,ang21,ang22,ang23,ang24,ang25,ang26,ang27,ang28')
    file_.write("\n")

count = 0
for mol_name in tqdm(test['molecule_name'].unique()):
    dat = read_xyz(mol_name)
    for atom1 in ob.OBMolAtomIter(dat['ob_mol']):
        for atom0 in ob.OBMolAtomIter(dat['ob_mol']):
            anglist = []
            for atom2 in ob.OBMolAtomIter(dat['ob_mol']):
                ang = atom0.GetAngle(atom1, atom2)
                ang /= 180
                anglist.append(ang)
            atom_0 = atom0.GetIdx() - 1
            atom_1 = atom1.GetIdx() - 1
            anglist.extend([0] * (max_size - len(anglist))) # Extend ang list
            anglist_commas = str(anglist)[1:-1]
            csv_line = f'{mol_name},{atom_0},{atom_1},{anglist_commas}'
            with open('angles_test.csv', mode='a') as file_:
                file_.write(csv_line)
                file_.write("\n")
#     if count == 100:
#         break
#     count += 1