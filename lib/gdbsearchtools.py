import re

def formatsmilesfile(file):
    ifile = open(file, 'r')
    contents = ifile.read()
    ifile.close()

    p = re.compile('([^\s]*).*\n')
    smiles = p.findall(contents)

    ofile = open(file, 'w')
    for mol in smiles:
        ofile.write(mol + '\n')
    ofile.close()

def get_symbols_rdkitmol(m):
    Na = m.GetNumAtoms()
    spc = []
    for a in range(Na):
        spc.append(m.GetAtomWithIdx(a).GetSymbol())
    return spc

#def conformation_generation:
