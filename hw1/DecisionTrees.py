import math
import copy
import random

# ------  Group member: Hao Yang, USCID 5284267375  --------

def xEntropy(p):
    if p == 0:
        return 0
    else:
        return -1 * p * math.log2(p)
def Entropy(I):                                 # calculate Entropy with list
    sum = 0
    for i in range(len(I)):
        sum += xEntropy(I[i])
    return sum
def Ent(D):                                     # calculate Entropy with Dataset(yes/no)
    yes = 0
    sum = len(D)
    for row in range(sum):
        if D[row][-1] == 'Yes':
            yes += 1
    I = [yes/sum, 1-(yes/sum)]
    return Entropy(I)
def EntDv(D,a,n):                               # Dataset D → Dnew which attri a == n
    Dnew = []
    for i in range(len(D)):
        if D[i][attriOper.index(a)] == n:
            Dnew.append(D[i])
    return Dnew
def Gain(D, a):                                 # D - dataset, a - string of attri
    record = []                                 # record number of each branch
    for i in range(attri_size[a]):
        record.append(0)
        for j in range(len(D)):
            o = attriOper.index(a)
            if D[j][attriOper.index(a)] == i:
                record[i] += 1                  # get |Dv|
    Infor = 0
    for k in range(len(record)):
        one = record[k]/sum(record)
        if one != 0:
            two = Ent(EntDv(D,a,k))
            Infor += one * two
    return Ent(D) - Infor
def more(D):
    YN = [0,0]
    for row in D:
        if row[-1] == 'Yes':
            YN[0] += 1
        else:
            YN[1] += 1
    flat = ''
    if YN[0] >= YN[1]:
        flat = 'Yes'
    else:
        flat = 'No'
    return flat
def Termin1(D):                                 # return 0 - all same class
    if len(D) <= 1:
        return 0
    flat = D[0][-1]
    for row in D[1:]:
        if row[-1] != flat:
            return 1
    return 0
def TreeGenerate(D,att):                        # D - dataset, att - attri list
    Tree = {}
    if Termin1(D) == 0 or len(att) == 0:        #if all in same class
        return more(D)

    infoGain_list = []
    for ele in att[:-1]:
        infoGain_list.append(Gain(D, ele))

    if max(infoGain_list) == 0:
        return more(D)
    locat = infoGain_list.index(max(infoGain_list))    # know which attri gains info most
    leng = attri_size[att[locat]]
    name = att[locat]
    Tree[name] = []
    for i in range(leng):                               # build subtree for all branch
        if len(EntDv(D,name,i)) == 0:
            Tree[name].append(more(D))
        else:
            Dn = copy.deepcopy(EntDv(D,name,i))
            for row in Dn:
                del row[locat]

            namelocat = att.index(name)
            att.remove(name)
            Tree[name].append(TreeGenerate(Dn,att))     # recurssive subdataset and subattribut
            att.insert(namelocat,name)
    return Tree
def Prediction(case, Tree):                     # get the result of prediction of case with Tree
    node_name = list(Tree.keys())[0]
    loc = attriOper.index(node_name)
    branch = dataMatrix_trans[node_name].index(case[loc])     # know case belongs to which branch on node
    new_Tree = Tree[node_name][branch]
    if new_Tree == 'Yes' or new_Tree == 'No':
        return new_Tree
    else:
        return Prediction(case, new_Tree)
def Bootstrapping(D,size,count):                       # bootstrappingly generate 'count' number of sub dataset with size 'size' 
    Dlist = []
    if len(D) <= count:
        return [D]
    for i in range(count):
        r = random.sample(range(len(D)),size)
        Dlist.append([])
        for j in range(len(r)):
            Dlist[i].append(D[r[j]])
    return Dlist
def Forest(D,att,size,count):                   # generate Forest
    Dlist = Bootstrapping(D,size,count)
    Tree_list = []
    for i in range(len(Dlist)):
        Tree_list.append(TreeGenerate(Dlist[i],att))
    return Tree_list
def Prediction_Forest(case,Treelist):
    result = []
    for i in range(len(Treelist)):
        result.append(Prediction(case,Treelist[i]))
    sumall = len(result)
    yes = 0
    for j in range(len(result)):
        if result[j] == 'Yes':
            yes += 1
    if yes >= sumall-yes:
        return 'Yes'
    else:
        return 'No'

# --------- main function from here ----------------
attri = []                                       # Attributes
dataMatrix = []                                  # raw data
with open('dt_data.txt','r') as data:
    line = data.readline()
    line = line.strip('()\n').replace(" ","").split(',')
    attri = line
    line = data.readline()                         #first line -- attribute
    while line:                                     # get data
        if line == '\n':                            # ignore 2nd line which is empty
            line = data.readline()
            continue
        line = line[4:].strip(';\n').replace(" ","").split(',')
        dataMatrix.append(line)
        line = data.readline()

#transfered txt into list data above

row = len(dataMatrix)
column = len(attri) - 1
attri_size = {}                                         #record the number of each terms
dataMatrix_index = copy.deepcopy(dataMatrix)
dataMatrix_trans = {}
for i in range(column):
    memory = []
    dataMatrix_trans[attri[i]] = []
    for j in range(row):
        if dataMatrix[j][i] not in memory:
            memory.append(dataMatrix[j][i])
            dataMatrix_trans[attri[i]].append(dataMatrix[j][i])
            dataMatrix_index[j][i] = memory.index(dataMatrix[j][i])
        else:
            dataMatrix_index[j][i] = memory.index(dataMatrix[j][i])
    attri_size[attri[i]] = len(memory)
dataOper = copy.deepcopy(dataMatrix_index)
attriOper = copy.deepcopy(attri)

#get attribute list and data matrix to operate above
result = {}
result = TreeGenerate(dataOper,attriOper)

test_case = ['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']                      #test case

predic = Prediction(test_case, result)                                                      #ID3 result

Forest = Forest(dataOper,attriOper,15,10)

predic2 = Prediction_Forest(test_case,Forest)                                               #ID3 random forest result

print(result)
print('Prediction of ID3 decision tree is:', predic)
print('Prediction of ID3 decision tree with random forest is:', predic2)