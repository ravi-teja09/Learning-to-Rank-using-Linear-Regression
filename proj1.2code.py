
# coding: utf-8

# In[359]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[360]:


maxAcc = 0.0
maxIter = 0
C_Lambda = 0.000005
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
n = 200 #multiplication factor in BigSigma i.e., in np.dot(n, BigSigma)
IsSynthetic = False


# In[361]:


# Creating a List of all the Target Values
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

# creating target values list for Training data. Training Data set to 80% of the total data 
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #from rawtarget which is a list of length of around 69,000
    t           = rawTraining[:TrainingLen] #will slice the list the integer value of TrainingLen
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize #end point of validation data
    t =rawData[TrainingCount+1:V_End] #slicing to get the validation set target samples
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# Creating data matrix from the specified csv file
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'r') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1) #removing features/columns with 0 variances (axis represents the dimensions where 0 stands for row and 1 stands for column)
    dataMatrix = np.transpose(dataMatrix)     
    #print ("Data Matrix Generated..")
    return dataMatrix

# Generating Training data (80% of Total samples)
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent)) #to determine the number of rows/samples in training data
    d2 = rawData[:,0:T_len] #slicing to get training data samples
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# Generating Validation data (10% of the data)
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01)) #10%
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

# Scaling the variance values using np.dot so that the values are not small enough to give uniform values. If the values are too small
# then all the variances will be too small and multiplication with other matrices will all generate nearly same values because all
# are small.

# Creating Variance matrix with only diagonal values or only variance of a particular feature but not the variance between
# different features.

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))#(41)
    DataT       = np.transpose(Data)#60k*41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))#60k*0.8=55k        
    varVect     = []
    for i in range(0,len(DataT[0])):#41
        vct = []
        for j in range(0,int(TrainingLen)):#50k
            vct.append(Data[i][j])    #1*50k:This is list that contains all the input in particular features
        varVect.append(np.var(vct))#41*41:Calculating variance here og features
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]#Removing co variance thats is variance between different features
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(n,BigSigma) #multiply with different values and see the effect on erms. We are increasing the value of variance
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow) # (x-mean)
    T = np.dot(BigSigInv,np.transpose(R)) # transpose of (x-mean) * inverse of variance matrix
    L = np.dot(R,T) #  (x-mean) * inverse of variance matrix * transpose of (x-mean)
    return L

# calculating a single feature value
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# generating the PHI/Feature matrix
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) #defining PHI matrix with zeros and dimensions = Training data samples x No. of clusters/basis fn.
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)): #for a particular column of the M features
        for R in range(0,int(TrainingLen)): # for each row
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv) #generating the FEATURE/PHI matrix
    #print ("PHI Generated..")
    return PHI


# finding/generating weights using Moore-Penrose pseudo inverse method including regularization
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0])) # identity matrix of 10*10
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda #creating identity matrix with lambda values
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR) #adding regularizer
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI) # calculating the multiplicative inverse
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

# predicting the target value
def GetValTest(VAL_PHI,W):
    Y = np.dot(VAL_PHI, W)
    ##print ("Test Out Generated..")
    return Y


# finding Root Mean Square error and Accuracy
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2) #summation of (Actual target - predicted target)^2
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]): #rounding the predicted target values to values with no decimal points and comparing with the actual target value
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[362]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[364]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[365]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[366]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[367]:


ErmsArr = []
AccuracyArr = []

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #Creating M number of clusters using K-means
Mu = kmeans.cluster_centers_ #### Coordinates of cluster centers

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[368]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[369]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W) #predicted target value for Training Data
VAL_TEST_OUT = GetValTest(VAL_PHI,W) #predicted the target value for Validation Data
TEST_OUT     = GetValTest(TEST_PHI,W) #predicting the target value for Testing Data

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[370]:


print ('UBITname      = rsunkara')
print ('Person Number = 50292191')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = " + str(M))
print("Lambda = " + str(C_Lambda))
print ("E_rms Training   = " + str(np.around(float(TrainingAccuracy.split(',')[1]), decimals = 4)))
print ("E_rms Validation = " + str(np.around(float(ValidationAccuracy.split(',')[1]), decimals = 4)))
print ("E_rms Testing    = " + str(np.around(float(TestAccuracy.split(',')[1]), decimals = 4)))
print ("Accuracy Training   = " + str(np.around(float(TrainingAccuracy.split(',')[0]), decimals = 4)))
print ("Accuracy Validation = " + str(np.around(float(ValidationAccuracy.split(',')[0]), decimals = 4)))
print ("Accuracy Testing    = " + str(np.around(float(TestAccuracy.split(',')[0]), decimals = 4)))


# ## Gradient Descent solution for Linear Regression

# In[371]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[372]:


W_Now        = np.dot(220, W) #random initialization, change 220 and see the accuracies
# when weights are already low, regularizer will decrease their effect even more and the fit will be very poor
#higher weights = better fit
La           = 10
learningRate = .01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []
L_Acc_TR = []
L_Acc_Val = []
L_Acc_Test = []
x1 = 0;
xn = 256; 

for i in range(x1,xn): #choose different number of samples and measure the erms, plot between them
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W) #adding regularizer
    Delta_W       = -np.dot(learningRate,Delta_E) #multiplying with learning rate
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    L_Acc_TR.append(float(Erms_TR.split(',')[0]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    L_Acc_Val.append(float(Erms_Val.split(',')[0]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    L_Acc_Test.append(float(Erms_Test.split(',')[0]))


# In[374]:


print ('----------Gradient Descent Solution--------------------')
print ("M = " + str(M))
print("Lambda = " + str(La/np.subtract(xn, x1))) # lambda is La/no. of samples
print("eta = " + str(learningRate))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

