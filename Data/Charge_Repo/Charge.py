#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:50:03 2020

@author: Ferran Domingo García

Código complementario al trabajo de fin de grado: IAC, Universidad de la Laguna. 2020. 
Dicho trabajo puede adquirirse contactando con la universidad, para facilitar la interpretación del código.

"""

import matplotlib.pyplot as plt
import torch
import io
import numpy as np
import matplotlib as plt

class ChargerIN ():

    def __init__(self):
            self.in_ = self.charge()
            self.files_ = ['Look at the SourceCode']

    def charge(self):
    
        f = open ('Charge_Repo/Presion.txt','r')
        t = open ('Charge_Repo/Temperature.txt','r')
        z = open ('Charge_Repo/Zeta.txt','r')

        ftext = f.read()
        ttext = t.read()
        ztext = z.read()

        ftext = ftext.split()
        ttext = ttext.split()
        ztext = ztext.split()

        f.close()
        t.close()
        z.close()

        MacroT = np.zeros((504,504,161))
        MacroP = np.zeros((504,504,161))
        MacroZ = np.zeros(161)

        for i in np.arange(161):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroT[k,j,i] = ttext[i+j+k]
        MacroT = torch.from_numpy(MacroT)            
        for i in np.arange(161):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroP[k,j,i] = ftext[i+j+k]
        MacroP = torch.from_numpy(MacroP) 
        for i in np.arange(161):
            MacroZ[i] = ztext[i]
        MacroZ = torch.from_numpy(MacroZ)
        return [MacroT.requires_grad_(True), MacroP.requires_grad_(True), MacroZ.requires_grad_(True)]

class ChargerOUT():

    def __init__(self):
        self.files_ = ['Look at the SourceCode']
        self.out_ = self.charge()

    def charge(self):
     
        e = open ('Charge_Repo/ElectronicPresure_tau.txt','r')
        ta = open ('Charge_Repo/tau.txt','r')
        T = open ('Charge_Repo/Temperature_tau.txt','r')

        etext = e.read()
        tatext = ta.read()
        Ttext = T.read()

        etext = etext.split()
        tatext = tatext.split()
        Ttext = Ttext.split()

        e.close()
        ta.close()
        T.close()

        MacroTA = np.zeros(86)
        Macroe= np.zeros((504,504,86))
        MacroTT = np.zeros((504,504,86))

        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroTT[k,j,i] = Ttext[i+j+k] 
        MacroTT =  torch.from_numpy(MacroTT)     
        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    Macroe[k,j,i] = etext[i+j+k]
        Macroe =  torch.from_numpy(Macroe)
        for i in np.arange(86):
            MacroTA[i] = tatext[i]
        MacroTA =  torch.from_numpy(MacroTA)
        
        return [MacroTT.requires_grad_(True),Macroe.requires_grad_(True), MacroTA.requires_grad_(True)]

class ChargerRest ():
    
    def __init__(self):
    
        self.files = ['./Charge_Repo/File_of_Variables/field.txt', './Charge_Repo/File_of_Variables/vel.txt', './Charge_Repo/File_of_Variables/incli.txt', './Charge_Repo/File_of_Variables/azimuth.txt', './Charge_Repo/File_of_Variables/pgas.txt', './Charge_Repo/File_of_Variables/pgas.txt', './Charge_Repo/File_of_Variables/rho.txt']
        self.columns = self.LoadTable()
            
    def LoadTable (self):
        
        x =list()
        count=0
        Macro = np.zeros((504,504,86))
        

        
        for i in self.files :
            
            #Linea de cogdigo para generar un micro .txt mediante un valor típico, vamos a tomarlo 1
            if count == 1 :
                 a= np.ones((504,504,86))*0
                 a = (torch.from_numpy(a))
                 x.append(a.requires_grad_(True))
            
            f = open (i)
            data = f.read()
            f.close()
            data = data.split()
    
            for s in np.arange(86):
                for j in np.arange(504):
                    for k in np.arange(504):
                        Macro[k][j][s] = data[k+j+s]
            
            count=count +1
            b = torch.from_numpy(Macro)
            x.append(b.requires_grad_(True))
        
        return x
  # Desire utiliza en la primera columna appended micro.txt por lo que debemos añadir un archivo:'./Charge_Repo/File_of_Variables/micro.txt',  
    
   
#Charge table has the structure of the columns tht desire programe needs for the input data ::::....      
class ChargerTABLE (object):
    
    def __init__(self, ChargerOUT,ChargerRest):
        
        self.ChOUT = ChargerOUT()
        self.ChAppend = ChargerRest()
        self.Table = self.fitData()
        
    def fitData (self):
        
        TensorList = list()
        for j in [2,0,1]:
            
            if j  == 2 :
               self.ChOUT.out_[j] = (self.ChOUT.out_[j].repeat(504*504,1)).view(-1,86)
               self.ChOUT.out_[j] = self.ChOUT.out_[j].view(504*504,86)
               TensorList.append(self.ChOUT.out_[j])
            else :
                self.ChOUT.out_[j] = self.ChOUT.out_[j].view(504*504,86)
                TensorList.append(self.ChOUT.out_[j])
            
        for i in np.arange(8):
            self.ChAppend.columns[i] = self.ChAppend.columns[i].view(504*504,86)
            TensorList.append(self.ChAppend.columns[i])
            
        for i in np.arange(len(TensorList)):
            TensorList[i] = TensorList[i].view(1,504*504,86)
            
        TensorList = torch.cat(TensorList[:], dim =0)
        TensorList = TensorList.chunk(504*504,dim=1)
        
        TensorList = list(TensorList)
        
        counter =0
        for k in TensorList:
            TensorList[counter]= torch.squeeze(k, dim = 1)
            counter=counter + 1 
            
        return TensorList   
"""
Notar aquí hemos supesto las profundidades ópticas las mismas para todos los pixeles
Ya que solo nos ha proporcionado eso, al igual que las Z, -- por lo que las P y T de cada modelo de pixel,
Deben de ser obtenidas leyendo dicho valor cuando este llega a dicho valor de pixel.
"""   
#Here manually you can tell the programe how many lines you want to store and to do not have
#to make more comprobartions insert manually the number to check in numberlines and LabelLines

class MakeTxt (object):
    
    def __init__ (self, table):
        
        self.Object = table.Table
        self.write()
       
    def line_prepender(self, filename):
        with open(filename, 'r+') as f:
            content = f.read().splitlines(True)
            content[0] = '0.0      1.0     0.0000000E+00\n'
            f.seek(0, 0)
            f.writelines(content)  
        
    def write(self) :
           
        #First all the tensors First need to convert to numpy arrays:
        #To be able to writ
        
        from astropy.io import ascii
        import copy
        
        #Then write pixel by pixel
        #for j in np.arange(list(self.Object[0].size())[0]):   
        for j in np.arange(len(self.Object)):  
            filename = './Desire_Repo/ModelCube/pixel' + str(j) + '.mod'
            
            ascii.write(np.transpose(self.Object[j].detach().numpy()), filename, overwrite=True)
            self.line_prepender(filename)

#Functionality to run 
def Desire ():
    
    import fileinput
    import os
    names = os.listdir('Desire_Repo/ModelCube/')
    bashCommand = "/Users/ferrannew/anaconda3/envs/pytorch_env/Proyectos_ML/desire/run/example/desire desire.dtrol"
   
    j = names[0].partition(".")[0]
    for i in names: 
        for line in fileinput.input("/Users/ferrannew/anaconda3/envs/pytorch_env/Proyectos_ML/desire/run/example/desire.dtrol"):
            # inside this loop the STDOUT will be redirected to the file
            # the comma after each print statement is needed to avoid double line breaks
            line.replace(j.partition(".")[0], i.partition(".")[0])
            j = i.partition(".")[0]
            os.system(bashCommand)

#Here make the changes to load the file for representation in type of ASCII table, which desire can read
#Change manually in the function pixel in order to store more pixles

class ReadTXT (self):
    return
    
#Here we supose that we will read the data from the ascii tables also;;
#Meaning we can train infinite models if given so ::::
    
class ChargerStoke ():
    
    def __init__(self):
        
        self.lines = self.open_()
        self.number_lines =3;
        self.label_lines = [4,23,24]
        
    def open_(self):
        
        fileO = open('./Desire_Repo/SokeCube/pixel0.per')
        read = fileO.read()
        fileO.close()
        read = read.split()
            
        keysCa = {'Line': '4' , 'landa' : [],  'I' : [] , 'U' : [], 'V' : [] , 'Q': []}
        keysFe = {'Line': '23','landa' : [], 'I' : [] , 'U' : [], 'V' : [] , 'Q': []}
        keysFeII = {'Line': '24', 'landa' : [],'I' : [] , 'U' : [], 'V' : [] , 'Q': []}
        keys = {'Ca' : keysCa, 'Fe' : keysFe, 'FeII' : keysFeII}
            
        count = 0
        for i in np.arange(len(read)/6,dtype=int):
            if keys['Ca']['Line'] == read[count]:
                keys['Ca']['landa'].append(float(read[count+1]))
                keys['Ca']['I'].append(float(read[count+2]))
                keys['Ca']['U'].append(float(read[count+3]))
                keys['Ca']['V'].append(float(read[count+4]))
                keys['Ca']['Q'].append(float(read[count+5]))
            else:        
                if keys['Fe']['Line'] == read[count]:
                    keys['Fe']['landa'].append(float(read[count+1]))
                    keys['Fe']['I'].append(float(read[count+2]))
                    keys['Fe']['U'].append(float(read[count+3]))
                    keys['Fe']['V'].append(float(read[count+4]))
                    keys['Fe']['Q'].append(float(read[count+5]))
                else:
                    keys['FeII']['Line'] == read[count]
                    keys['FeII']['landa'].append(float(read[count+1]))
                    keys['FeII']['I'].append(float(read[count+2]))
                    keys['FeII']['U'].append(float(read[count+3]))
                    keys['FeII']['V'].append(float(read[count+4]))
                    keys['FeII']['Q'].append(float(read[count+5]))
            count = count+6
        return keys
    
    def paint(self):
        
        names = list(self.lines)
        variables = list(self.lines[names[0]])
        
        for i in names:
            fig, ax = plt.subplots(2,2)

            ax[0,0].set_title(i + '  ' + variables[2])
            ax[0,1].set_title(i + '  ' + variables[3])
            ax[1,0].set_title(i + '  ' + variables[4])
            ax[1,1].set_title(i + '  ' + variables[5])
            ax[0,0].scatter(self.lines[i][variables[1]],self.lines[i][variables[2]])
            ax[0,1].scatter(self.lines[i][variables[1]],self.lines[i][variables[3]])
            ax[1,0].scatter(self.lines[i][variables[1]],self.lines[i][variables[4]])
            ax[1,1].scatter(self.lines[i][variables[1]],self.lines[i][variables[5]])
            
            ax[0,0].set_xticks([]) 
            ax[0,0].set_yticks([])
            ax[0,0].set_xlabel('landa') 
            ax[0,0].set_ylabel(variables[2])
            
            ax[1,1].set_xticks([]) 
            ax[1,1].set_yticks([])
            ax[1,1].set_xlabel('landa') 
            ax[1,1].set_ylabel(variables[5])
            
            ax[1,0].set_xticks([]) 
            ax[1,0].set_yticks([])
            ax[1,0].set_xlabel('landa') 
            ax[1,0].set_ylabel(variables[4])
            
            ax[0,1].set_xticks([]) 
            ax[0,1].set_yticks([])
            ax[0,1].set_xlabel('landa') 
            ax[0,1].set_ylabel(variables[3])
  #Class to visualize a .per concrete, but not charge the whole macro    
            
            
class ChargeMStokes ():
        #We will use a 504,504 simulations , but can be changed manually:
     
    def __init__(self):
        self.Macro = self.Charge()
           
    def Charge (self):
        
        TensorList = list()
        SubtensorList = list()
        
        for i in os.listdir('./Desire_Repo/StokeCube/'):
            with open('./Desire_Repo/StokeCube/'+i, 'r') as f:   
                
                read = f.read()
                f.close()
                read = read.split()
                   
                #AA Priori no sabemos como de grande va a ser el mallado asi que la única solución
                #Es emplear una estructura de listas encadenadas
                
                landa = list()
                Q = list()
                V = list()
                U = list()
                I = list()
                
                count = 0
                    
                while '4' == read[count] :
                    landa.append(float(read[count+1]))
                    I.append(float(read[count+2]))
                    U.append(float(read[count+3]))
                    V.append(float(read[count+4]))
                    Q.append(float(read[count+5]))
                    count = count+6
                    
                SubtensorList.append(copy.deepcopy([landa,I,U,V,Q]))
                I.clear();Q.clear();U.clear();V.clear();landa.clear();
               
                while '23' == read[count]:
                    landa.append(float(read[count+1]))
                    I.append(float(read[count+2]))
                    U.append(float(read[count+3]))
                    V.append(float(read[count+4]))
                    Q.append(float(read[count+5]))
                    count = count+6
                    
                SubtensorList.append(copy.deepcopy([landa,I,U,V,Q]))
                I.clear();Q.clear();U.clear();V.clear();landa.clear();
                    
                while '24' == read[count] :
                    landa.append(float(read[count+1]))
                    I.append(float(read[count+2]))
                    U.append(float(read[count+3]))
                    V.append(float(read[count+4]))
                    Q.append(float(read[count+5]))
                    count = count+6
                    
                    if count == int(len(read)):
                        break
                        
                SubtensorList.append(copy.deepcopy([landa,I,U,V,Q]))
                I.clear();Q.clear();U.clear();V.clear();landa.clear();
                
                for j in np.arange(len(SubtensorList)):
                    SubtensorList[j] = torch.tensor(SubtensorList[j] , dtype = float, requires_grad =True) 
           
            TensorList.append(SubtensorList)
            
        return TensorList
  #Returns a list of pixels with, inside each pixel each line is a tensor which has to be trained

  
class Load_MStokes_MTable(object):
    
    def __init__(self, Table, Stoke):
        
        self.MSimulation = Table.table
        self.MStoke = Stoke.Macro
        #Make uso of 3 lines with possible different spacial gradings
        self.lines = lines()
        
    def lines (self):
        
        lines = list()
        
        counter  =0;
        for i in len(self.MStoke[0]):
            a = list(self.MStoke[0][i].size())[-1]
            lines.append(LinearStokes(5*a, 86*11)) 
        
        return lines
                          
    def train (self):
        
        lines = list();   
        a = np.minimum(len(self.Simulation),len(self.MStoke))
        
        for epoch in np.arange(40):
            
            for i in np.arange(a/252):
                #Here we suppose we charge the same amount of lines
                for j in np.arange(len(a[0])):
                    
                    batchTrain = self.MSimulation[252*i:(i+1)*252 -1]
                    batchTest = self.MStoke[252*i:(i+1)*252 -1][j]
                    
                if i == ((504*504)-1):
                    break
                            
            for j in len(self.MStoke[i])
                lines.append(self.MStoke[i][j])
           
#Remeber that has to be passed the table object already create and the object Stoke, 
#Already created, with one to one identification in :
            
class LinearStokes(nn.Module,in_,out_):
   
    def __init__(self,in_,out_):
        super(LinearStokes,self).__init__()
        self.fc1 = nn.Linear(in_,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,100)
        self.fc4 = nn.Linear(100,out_)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x







"""
Class used to store in colums the tensors corresponfing to the fiels variables, in order to train neural networks
""" 
class Model_Reader():
    
    def __init__(self):  
        with open("/Charge_Repo/Modelpxp.pkl","r") as f:
            self.pxp =pickle.load( f)
        with open("/Charge_Repo/Modelpack.pkl","r") as f:
            self.pack = pickle.load( f)

"""
"""    
def update_Model (Model_Reader_Object, Loader_Object):
    
    try:
        self.pxp
    try:
        self.pack
"""                
"""
Class used to train, sintetic models: For that the ntework has to be loades with the correct parametes, once it converge for the training set::

Option 1:
Once trined this class sintetize artificial TAU,E.Pressure and Temperature and attaches them to the cubes
stored in LoadTable / which concatene columns to generate full cubes to compare to numercal values for:  

Option 2:
It uses real data for the training the TAU,Press,Temperature and attach them to the cubes:
    
Option 3: Comprares the results once run Option 1 and Option 2.
""" 
 

"A la classe los objetos se le tienen que pasar inicializados ya - tal que la designación en OB refiere a que el objeto de la calse ya esta creado " 
"Por ahora: testing de la tabla se refiere a muestras reales, que corresponden exatamente a las columnas de LoadSintetic, que se adjuntan a los valores de Pe,Tau,T"
"para formar una tabla con los que ejecutar desire para la simulación nmérica"
"Llamaremos LoadSimulation a los parámetros generados con el programa desire, no confundir con LoadSintetic "

"""
class Sintesis_Training (object):

    def __init__(self, LoadModuleCL_OB,, LoadSintetic_OB , LoadSimulation_OB, Convolutional_TPT):
        
        for i in LoadModuleCL_OB.testing
            LoadSintetic_OB.columns.append(i)
        
        self.filesTable = LoadSintetic_OB.files()
        self.filesTest = LoadSimulation_OB.columns()   
        
        self.Table = LoadSintetic_OB.columns      
        self.Testing = LoadSimulation_OB.columns
        
        self.module = Convolutional_TPT()
        self.netD = {}
        self.losses = list()
    
    def train (self,Convolutional_TPT) :
        
        count =0;
         for i in self.Table:
            self.Table[count] = i.view(-1,86)
            count = count + 1 
            
         for epoch in np.arange(504*504):

            packTest = torch.cat((self.Table[0][epoch][:], self.Table[1][epoch][:],self.Table[2][epoch][:], self.Table[3][epoch][:],self.Table[4][epoch][:],self.Table[5][epoch][:],self.Table[6][epoch][:],self.Table[7][epoch][:],self.Table[8][0][:] ),1)
            packTrain = torch.cat((self.training[0][epoch][:], self.testing[1][epoch][:],self.training[2][0][:]),1)
               
            
        self.netDict.clear()
        self.optimDict.clear()

        for epoch in np.arange(504*504):

"""           
        
        