import pandas as pd
import numpy as np
import os
import csv
import openbabel
from tqdm import tqdm
import gc
obConversion = openbabel.OBConversion()
_ = obConversion.SetInAndOutFormats("xyz", "mol2")

file_list = [x for x in os.listdir('../data/FE015') if 'FE015' in x]

for filen in file_list:
    print('Running for {}'.format(filen))
    print('Checking to see is this is created {}'.format(filen.replace('FE015','FE016')))
    if filen.replace('FE015','FE016') in os.listdir('../data/FE016'):
        print('Already done')
        continue
    file_short = filen.split('.')[0][6:]
    df = pd.read_parquet('../data/FE015/' + filen)
    angle_list_clos_2nd = list(zip(df['id'], df['molecule_name'],
                                     df['atom_index_1'],df['atom_index_0'],df['6th_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['7th_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['8th_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['9th_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['10th_closest_to_0'],
                                     df['atom_index_1'],df['atom_index_0'],df['11th_closest_to_0'],
                                     df['atom_index_0'],df['atom_index_1'],df['6th_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['7th_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['8th_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['9th_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['10th_closest_to_1'],
                                     df['atom_index_0'],df['atom_index_1'],df['11th_closest_to_1'],
                                     df['atom_index_0'],df['closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['2nd_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['3rd_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['4th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['5th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['6th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['7th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['8th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['9th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['10th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['11th_closest_to_0'],df['atom_index_1'],
                                     df['atom_index_0'],df['closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['2nd_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['3rd_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['4th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['5th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['6th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['7th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['8th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['9th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['10th_closest_to_1'],df['atom_index_1'],
                                     df['atom_index_0'],df['11th_closest_to_1'],df['atom_index_1'],
                                  ))
    with open(r'./FE16-temp/angle_{}.csv'.format(file_short), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','molecule_name',
                         'angle_1_0_6th0', 'angle_1_0_7th0', 'angle_1_0_8th0', 'angle_1_0_9th0', 'angle_1_0_10th0', 'angle_1_0_11th0',
                         'angle_1_0_6th1', 'angle_1_0_7th1', 'angle_1_0_8th1', 'angle_1_0_9th1', 'angle_1_0_10th1', 'angle_1_0_11th1',
                         'angle_0_closest0_1', 'angle_0_2nd0_1', 'angle_0_3rd0_1', 'angle_0_4th0_1', 'angle_0_5th0_1',
                         'angle_0_6th0_1', 'angle_0_7th0_1', 'angle_0_8th0_1', 'angle_0_9th0_1', 'angle_0_10th0_1','angle_0_11th0_1',
                         'angle_0_closest1_1', 'angle_0_2nd1_1', 'angle_0_3rd1_1', 'angle_0_4th1_1', 'angle_0_5th1_1',
                         'angle_0_6th1_1', 'angle_0_7th1_1', 'angle_0_8th1_1', 'angle_0_9th1_1', 'angle_0_10th1_1','angle_0_11th1_1'])
    for mylist in tqdm(angle_list_clos_2nd):
        mol = openbabel.OBMol()
        id_name = mylist[0]
        mol_name = mylist[1]
        idx = 2
        angle_list = ['angle_1_0_6th0', 'angle_1_0_7th0', 'angle_1_0_8th0', 'angle_1_0_9th0', 'angle_1_0_10th0', 'angle_1_0_11th0',
                         'angle_1_0_6th1', 'angle_1_0_7th1', 'angle_1_0_8th1', 'angle_1_0_9th1', 'angle_1_0_10th1', 'angle_1_0_11th1',
                         'angle_0_closest0_1', 'angle_0_2nd0_1', 'angle_0_3rd0_1', 'angle_0_4th0_1', 'angle_0_5th0_1',
                         'angle_0_6th0_1', 'angle_0_7th0_1', 'angle_0_8th0_1', 'angle_0_9th0_1', 'angle_0_10th0_1','angle_0_11th0_1',
                         'angle_0_closest1_1', 'angle_0_2nd1_1', 'angle_0_3rd1_1', 'angle_0_4th1_1', 'angle_0_5th1_1',
                         'angle_0_6th1_1', 'angle_0_7th1_1', 'angle_0_8th1_1', 'angle_0_9th1_1', 'angle_0_10th1_1','angle_0_11th1_1']
        ang_list = []
        for ang in angle_list:
            try:
                #angle_clos_0_2nd
                a = int(mylist[idx]+1)
                b = int(mylist[idx+1]+1)
                c = int(mylist[idx+2]+1)
                obConversion.ReadFile(mol, '../input/structures/{}.xyz'.format(mol_name))
                a_atom = mol.GetAtom(a)
                b_atom = mol.GetAtom(b)
                c_atom = mol.GetAtom(c)
                ang_calced = b_atom.GetAngle(a_atom, c_atom)
            except ValueError:
                ang_calced = np.nan
#             print(a, b, c)
#             print(ang_calced)
            ang_list.append(ang_calced)
#             print(ang_list)
            idx += 3
        with open(r'./FE16-temp/angle_{}.csv'.format(file_short), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id_name,
                                     mol_name,
                                     ang_list[0],
                                     ang_list[1],
                                     ang_list[2],
                                     ang_list[3],
                                     ang_list[4],
                                     ang_list[5],
                                     ang_list[6],
                                     ang_list[7],
                                     ang_list[8],
                                     ang_list[9],
                                     ang_list[10],
                                     ang_list[11],
                                     ang_list[12],
                                     ang_list[13],
                                     ang_list[14],
                                     ang_list[15],
                                     ang_list[16],
                                     ang_list[17],
                                     ang_list[18],
                                     ang_list[19],
                                     ang_list[20],
                                     ang_list[21],
                                     ang_list[22],
                                     ang_list[23],
                                     ang_list[24],
                                     ang_list[25],
                                     ang_list[26],
                                     ang_list[27],
                                     ang_list[28],
                                     ang_list[29],
                                     ang_list[30],
                                     ang_list[31],
                                     ang_list[32],
                                     ang_list[33],
                                    ])
    angle_df = pd.read_csv('./FE16-temp/angle_{}.csv'.format(file_short))
    df['angle_1_0_6th0'] = angle_df['angle_1_0_6th0']
    df['angle_1_0_7th0'] = angle_df['angle_1_0_7th0']
    df['angle_1_0_8th0'] = angle_df['angle_1_0_8th0']
    df['angle_1_0_9th0'] = angle_df['angle_1_0_9th0']
    df['angle_1_0_10th0'] = angle_df['angle_1_0_10th0']
    df['angle_1_0_11th0'] = angle_df['angle_1_0_11th0']
    df['angle_1_0_6th1'] = angle_df['angle_1_0_6th1']
    df['angle_1_0_7th1'] = angle_df['angle_1_0_7th1']
    df['angle_1_0_8th1'] = angle_df['angle_1_0_8th1']
    df['angle_1_0_9th1'] = angle_df['angle_1_0_9th1']
    df['angle_1_0_10th1'] = angle_df['angle_1_0_10th1']
    df['angle_1_0_11th1'] = angle_df['angle_1_0_11th1']
    df['angle_0_closest0_1'] = angle_df['angle_0_closest0_1']
    df['angle_0_2nd0_1'] = angle_df['angle_0_2nd0_1']
    df['angle_0_3rd0_1'] = angle_df['angle_0_3rd0_1']
    df['angle_0_4th0_1'] = angle_df['angle_0_4th0_1']
    df['angle_0_5th0_1'] = angle_df['angle_0_5th0_1']
    df['angle_0_6th0_1'] = angle_df['angle_0_6th0_1']
    df['angle_0_7th0_1'] = angle_df['angle_0_7th0_1']
    df['angle_0_8th0_1'] = angle_df['angle_0_8th0_1']
    df['angle_0_9th0_1'] = angle_df['angle_0_9th0_1']
    df['angle_0_10th0_1'] = angle_df['angle_0_10th0_1']
    df['angle_0_11th0_1'] = angle_df['angle_0_11th0_1']
    df['angle_0_closest1_1'] = angle_df['angle_0_closest1_1']
    df['angle_0_2nd1_1'] = angle_df['angle_0_2nd1_1']
    df['angle_0_3rd1_1'] = angle_df['angle_0_3rd1_1']
    df['angle_0_4th1_1'] = angle_df['angle_0_4th1_1']
    df['angle_0_5th1_1'] = angle_df['angle_0_5th1_1']
    df['angle_0_6th1_1'] = angle_df['angle_0_6th1_1']
    df['angle_0_7th1_1'] = angle_df['angle_0_7th1_1']
    df['angle_0_8th1_1'] = angle_df['angle_0_8th1_1']
    df['angle_0_9th1_1'] = angle_df['angle_0_9th1_1']
    df['angle_0_10th1_1'] = angle_df['angle_0_10th1_1']
    df['angle_0_11th1_1'] = angle_df['angle_0_11th1_1']

    df.to_parquet('../data/FE016/' + filen.replace('FE015','FE016'))
    del df
    del mol
    del angle_list_clos_2nd
    gc.collect()
