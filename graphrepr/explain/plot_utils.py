from collections import namedtuple

def make_list(length, prefix):
    return [f'{prefix} {i}' for i in range(length)]


our_atoms = ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
our_neighs = make_list(6, 'N')  # max 5 neighbours
our_hydro = make_list(5, 'H')   # max 4 hydrogens
charge = 'charge'
ringness = 'is in a ring'
aroma = 'is aromatic'
hybri=['sp', 'sp2', 'sp3', 'sp3d', 'sp3d2']

cheminet_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Br ', 'I', 'other']
duvenaud_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'other']

Kolorex = namedtuple("Kolorex", ["name", "start", "end", "colour"])

repr_1 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('neighbours', 11, 17, 'tab:orange'),
          Kolorex('hydrogens', 17, 22, 'tab:green'),
          Kolorex('charge', 22, 23, 'tab:red'),
          Kolorex('ringness', 23, 24, 'tab:purple'),
          Kolorex('aromaticity', 24, 25, 'tab:brown')
          ], our_atoms + our_neighs + our_hydro + [charge, ringness, aroma]

repr_2 = [Kolorex('atom type', 0, 11, 'tab:blue'), ], our_atoms

repr_3 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('neighbours', 11, 17, 'tab:orange')
          ], our_atoms + our_neighs

repr_4 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('hydrogens', 11, 16, 'tab:green'),
          ], our_atoms + our_hydro

repr_5 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('charge', 11, 12, 'tab:red'),
          ], our_atoms + [charge, ]

repr_6 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('ringness', 11, 12, 'tab:purple'),
          ], our_atoms + [ringness, ]

repr_7 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('aromaticity', 11, 12, 'tab:brown')
          ], our_atoms + [aroma, ]

repr_8 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('hydrogens', 11, 16, 'tab:green'),
          Kolorex('charge', 16, 17, 'tab:red'),
          Kolorex('ringness', 17, 18, 'tab:purple'),
          Kolorex('aromaticity', 18, 19, 'tab:brown')
          ], our_atoms + our_hydro + [charge, ringness, aroma]

repr_9 = [Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('neighbours', 11, 17, 'tab:orange'),
          Kolorex('charge', 17, 18, 'tab:red'),
          Kolorex('ringness', 18, 19, 'tab:purple'),
          Kolorex('aromaticity', 19, 20, 'tab:brown')
          ], our_atoms + our_neighs + [charge, ringness, aroma]

repr_10 =[Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('neighbours', 11, 17, 'tab:orange'),
          Kolorex('hydrogens', 17, 22, 'tab:green'),
          Kolorex('ringness', 22, 23, 'tab:purple'),
          Kolorex('aromaticity', 23, 24, 'tab:brown')
          ], our_atoms + our_neighs + our_hydro + [ringness, aroma]

repr_11 =[Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('neighbours', 11, 17, 'tab:orange'),
          Kolorex('hydrogens', 17, 22, 'tab:green'),
          Kolorex('charge', 22, 23, 'tab:red'),
          Kolorex('aromaticity', 23, 24, 'tab:brown')
          ], our_atoms + our_neighs + our_hydro + [charge, aroma]

repr_12 =[Kolorex('atom type', 0, 11, 'tab:blue'),
          Kolorex('neighbours', 11, 17, 'tab:orange'),
          Kolorex('hydrogens', 17, 22, 'tab:green'),
          Kolorex('charge', 22, 23, 'tab:red'),
          Kolorex('ringness', 23, 24, 'tab:purple'),
          ], our_atoms + our_neighs + our_hydro + [charge, ringness]

cheminet=[Kolorex('atom type', 0, 23, 'tab:blue'),
          Kolorex('vdW radius', 23, 24, 'tab:gray'),
          Kolorex('covalent radius', 24, 25, 'slategray'),
          Kolorex('ringness', 25, 31, 'tab:purple'),
          Kolorex('aromaticity', 31, 32, 'tab:brown'),
          Kolorex('charge', 32, 33, 'tab:red'),
          ], cheminet_atoms + ['vdW radius', 'covalent radius'] + [f'is in a ring size {i}' for i in range(3, 9)] + [aroma, charge]

dpchmstb=[Kolorex('atom type', 0, 13, 'tab:blue'),
          Kolorex('degree', 13, 19, 'tab:orange'),
          Kolorex('hydrogens', 19, 24, 'tab:green'),
          Kolorex('valence', 24, 30, 'tab:cyan'),
          Kolorex('aromaticity', 30, 31, 'tab:brown'),
          Kolorex('charge', 31, 32, 'tab:red'),
          Kolorex('radical electrons', 32, 33, 'tab:olive'),
          Kolorex('hybridisation', 33, 38, 'tab:pink'),
          Kolorex('Gesteiger charge', 38, 39, 'lightcoral')
          ], ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H',  'other'] + make_list(6, 'D') + make_list(5, 'H') + make_list(6, 'V') + [aroma, charge, '# radical electrons'] + hybri + ['Gasteiger charge']
# D stands for degree (rdkit's GetDegree)

duvenaud=[Kolorex('atom type', 0, 44, 'tab:blue'),
          Kolorex('degree', 44, 50, 'tab:orange'),
          Kolorex('hydrogens', 50, 55, 'tab:green'),
          Kolorex('valence', 55, 61, 'tab:cyan'),
          Kolorex('aromaticity', 61, 62, 'tab:brown'),
          ], duvenaud_atoms + make_list(6, 'D') + make_list(5, 'H') + make_list(6, 'V') + [aroma, ]

dmpnn  = [Kolorex('atom type', 0, 101, 'tab:blue'),
          Kolorex('degree', 101, 108, 'tab:orange'),
          Kolorex('charge', 108, 114, 'tab:red'),
          Kolorex('chirality', 114, 119, 'tab:cyan'),
          Kolorex('hydrogens', 119, 125, 'tab:green'),
          Kolorex('hybridisation', 125, 131, 'tab:pink'),
          Kolorex('aromaticity', 131, 132, 'tab:brown'),
          Kolorex('mass', 132, 133, 'tab:purple')
          ], make_list(101, 'A') + make_list(6, 'TD') + ['TD other'] + ['-1', '-2', '1', '2', '0', 'other'] + make_list(4, 'C') + ['C other'] + make_list(5, 'H') + ['H other'] + hybri + ['hy other'] + [aroma, 'mass']
# TD stands for total degree (rdkit's GetTotalDegree)

kolorex = {'1-repr': repr_1, 
           '2-repr': repr_2,
           '3-repr': repr_3,
           '4-repr': repr_4,
           '5-repr': repr_5,
           '6-repr': repr_6,
           '7-repr': repr_7,
           '8-repr': repr_8,
           '9-repr': repr_9,
           '10-repr': repr_10,
           '11-repr': repr_11,
           '12-repr': repr_12,
           'cheminet-repr': cheminet,
           'deepchemstable-repr': dpchmstb,
           'dmpnn-repr': dmpnn,
           'duvenaud-repr':duvenaud
          }
