"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""

import numpy as np



class mlp:
    def __init__(self, inputs, targets, nhidden):
        #set up neural networks
        self.beta = 1 
        self.eta = 0.1 #learning rates 
        self.momentum = 0.9 #optional
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin) #41,12
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)#13,8
        
    
  # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):
        
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            self.train(inputs,targets,niterations=100)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.forward(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print ("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error

    def train(self, inputs, targets,niterations=100):
        #defining inputs with bias
        change = list (range(self.ndata))
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        #change = range(inputs)
        # variables for weights updating 
        updatewh = np.zeros((np.shape(self.weights1)))
        updatewo = np.zeros((np.shape(self.weights2)))
        #using sequential traning method 
        for n in range(niterations):
            self.outputs = self.forward(inputs)
            error1 = 0.5*np.sum((self.outputs-targets)**2)
            if ((n%10)==0):
                print ('Epoch:', n , 'Error:',error1)
            deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                         
            updatewh = self.eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatewh
            updatewo = self.eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatewo
            self.weights1 -= updatewh
            self.weights2 -= updatewo
        np.random.shuffle(change)
        inputs = inputs[change,:]
        targets = targets[change,:]
        
                    
    def forward(self, inputs):
        self.hidden = np.dot(inputs,self.weights1);
      
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = np.dot(self.hidden,self.weights2);
        outputs = 1.0/(1.0+np.exp(-self.beta*outputs))
        return outputs
            
                         

    def confusion(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.forward(inputs)
        
        nclasses = np.shape(targets)[1]
      

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print ("Confusion matrix is:" )
        print (cm)
        print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
        #great metrix: correct matrix, see time, how correct your network printing matrix  

