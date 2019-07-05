import pandas as pd
import numpy as np
import os
import csv
import openbabel
from tqdm import tqdm
obConversion = openbabel.OBConversion()
_ = obConversion.SetInAndOutFormats("xyz", "mol2")

file_list = [x for x in os.listdir('../data/') if 'FE012' in x]

for filen in file_list:
    print('Running for {}'.format(filen))
    file_short = filen.split('.')[0][6:]
    df = pd.read_parquet('../data/' + filen)
    angle_list_clos_2nd = list(zip(df['id'], df['molecule_name'],
                                     df['closest_to_0'], df['atom_index_0'], df['2nd_closest_to_0'],
                                     df['closest_to_1'],df['atom_index_1'],df['2nd_closest_to_1'],
                                     df['atom_index_1'],df['atom_index_0'],df['closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['2nd_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['3rd_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['4th_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['5th_closest_to_0'],
                                     df['atom_index_0'],df['atom_index_1'],df['closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['2nd_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['3rd_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['4th_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['5th_closest_to_1'],
                                  ))
    with open(r'./FE13-temp/angle_{}.csv'.format(file_short), 'w') as f:
        print('running')
        writer = csv.writer(f)
        writer.writerow(['id','molecule_name','angle_clos_0_2nd','angle_clos_1_2nd',
                         'angle_1_0_closest0', 'angle_1_0_2nd0', 'angle_1_0_3rd0', 'angle_1_0_4th0', 'angle_1_0_5th0',
                         'angle_0_1_closest1', 'angle_0_1_2nd1', 'angle_0_1_3rd1', 'angle_0_1_4th1', 'angle_0_1_5th1'])
    for mylist in tqdm(angle_list_clos_2nd):
        mol = openbabel.OBMol()
        id_name = mylist[0]
        mol_name = mylist[1]
        a_0 = mylist[2]+1
        b_0 = mylist[3]+1
        c_0 = mylist[4]+1
        obConversion.ReadFile(mol, '../input/structures/{}.xyz'.format(mol_name))
        a_atom_0 = mol.GetAtom(a_0)
        b_atom_0 = mol.GetAtom(b_0)
        c_atom_0 = mol.GetAtom(c_0)

        a_1 = mylist[5]+1
        b_1 = mylist[6]+1
        c_1 = mylist[7]+1
        a_atom_1 = mol.GetAtom(a_1)
        b_atom_1 = mol.GetAtom(b_1)
        c_atom_1 = mol.GetAtom(c_1)

        a_2 = mylist[8]+1
        b_2 = mylist[9]+1
        c_2 = mylist[10]+1
        a_atom_2 = mol.GetAtom(a_2)
        b_atom_2 = mol.GetAtom(b_2)
        c_atom_2 = mol.GetAtom(c_2)

        a_3 = mylist[11]+1
        b_3 = mylist[12]+1
        c_3 = mylist[13]+1
        a_atom_3 = mol.GetAtom(a_3)
        b_atom_3 = mol.GetAtom(b_3)
        c_atom_3 = mol.GetAtom(c_3)

        a_4 = mylist[14]+1
        b_4 = mylist[15]+1
        c_4 = mylist[16]+1
        a_atom_4 = mol.GetAtom(a_4)
        b_atom_4 = mol.GetAtom(b_4)
        c_atom_4 = mol.GetAtom(int(c_4))

        a_5 = mylist[17]+1
        b_5 = mylist[18]+1
        c_5 = mylist[19]+1
        if type(c_5) is np.nan:
            print('its nan')
        a_atom_5 = mol.GetAtom(a_5)
        b_atom_5 = mol.GetAtom(b_5)
        c_atom_5 = mol.GetAtom(int(c_5))

        a_6 = mylist[20]+1
        b_6 = mylist[21]+1
        c_6 = mylist[22]+1
        a_atom_6 = mol.GetAtom(a_6)
        b_atom_6 = mol.GetAtom(b_6)
        c_atom_6 = mol.GetAtom(c_6)


        with open(r'./FE13-temp/angle_{}.csv'.format(file_short), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id_name,
                                     mol_name,
                                     b_atom_0.GetAngle(a_atom_0, c_atom_0),
                                     b_atom_1.GetAngle(a_atom_1, c_atom_1),
                                     b_atom_2.GetAngle(a_atom_2, c_atom_2), #mol.GetAtom(mylist[9]+1).GetAngle(mol.GetAtom(mylist[8]), mol.GetAtom(mylist[10])), # angle_1_0_closest0
                                     b_atom_3.GetAngle(b_atom_3, c_atom_3), # mol.GetAtom(mylist[12]+1).GetAngle(mol.GetAtom(mylist[11]), mol.GetAtom(mylist[13])), # angle_1_0_2nd0
                                     b_atom_4.GetAngle(b_atom_4, c_atom_4), # mol.GetAtom(mylist[15]+1).GetAngle(mol.GetAtom(mylist[14]), mol.GetAtom(mylist[16])), # angle_1_0_3rd0
                                     b_atom_5.GetAngle(b_atom_5, c_atom_5), # mol.GetAtom(mylist[18]+1).GetAngle(mol.GetAtom(mylist[17]), mol.GetAtom(mylist[19])), # angle_1_0_4th0
                                     b_atom_6.GetAngle(b_atom_6, c_atom_6), # mol.GetAtom(mylist[21]+1).GetAngle(mol.GetAtom(mylist[20]), mol.GetAtom(mylist[22])), # angle_1_0_5th0
                                     # mol.GetAtom(mylist[24]+1).GetAngle(mol.GetAtom(mylist[23]), mol.GetAtom(mylist[25])), # angle_0_1_closest1
                                     # mol.GetAtom(mylist[27]+1).GetAngle(mol.GetAtom(mylist[26]), mol.GetAtom(mylist[28])), # angle_0_1_2nd1
                                     # mol.GetAtom(mylist[30]+1).GetAngle(mol.GetAtom(mylist[29]), mol.GetAtom(mylist[31])), # angle_0_1_3rd1
                                     # mol.GetAtom(mylist[33]+1).GetAngle(mol.GetAtom(mylist[32]), mol.GetAtom(mylist[34])), # angle_0_1_4th1
                                     # mol.GetAtom(mylist[36]+1).GetAngle(mol.GetAtom(mylist[35]), mol.GetAtom(mylist[37])), # angle_0_1_5th1
                                    ])
    angle_df = pd.read_csv('./FE13-temp/angle_{}.csv'.format(file_short))
    df['angle_clos_0_2nd'] = angle_df['angle_clos_0_2nd']
    df['angle_clos_1_2nd'] = angle_df['angle_clos_1_2nd']
    df.to_parquet('../data/' + filen.replace('FE012','FE013'))
