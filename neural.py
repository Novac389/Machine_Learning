import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

class Neural:
    def __init__(self, 
                 layers, 
                 epochs=128, 
                 learning_rate = 0.001,
                 batch_size=32,
                 hidden_function= "relu",
                 output_function = "linear",
                 init_method = "rand",
                 optimizer = "nesterov",
                 momentum = 0.9,
                 momentum_schedule = False,
                 beta1=0.9, beta2=0.999,
                 verbose=1,
                 l2_lambda = 0.01,
                 patience = 10,
                 early_stopping = True):

        self.layer_structure = layers 
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_function = getattr(self,"_"+self.__class__.__name__+"__"+hidden_function)
        self.output_function = getattr(self,"_"+self.__class__.__name__+"__"+output_function)
                     
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
                     
        self.optimizer = optimizer
        self.momentum = momentum
        self.momentum_schedule = momentum_schedule

        self.early_stopping = early_stopping
        self.patience = patience

        self.verbose = verbose
                     
        self.prev_weights = None
        self.init_method = init_method   
        self.weight_bias = self.init_weights()
                     
        self.losses = {"train": [], "validation": []}
        self.accuracy = {"train": [], "validation": []}
        
        # Calculate the total number of weights and biases
        w = sum(self.layer_structure[i] * self.layer_structure[i + 1] for i in range(len(self.layer_structure) - 1))
        b = sum(self.layer_structure[i + 1] for i in range(len(self.layer_structure) - 1))

        #inizialize last change arrays used in momentum
        self.last_change_W = np.zeros(w)
        self.last_change_b = np.zeros(b)
        
        #parameters of adam optimizer            
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(w+b)
        self.v = np.zeros(w+b)
        
        #number of iteration
        self.t = 0

        #parameters for momentum_schedule
        self.a = 0.0
        self.a_prev = 1.0
        
    def fit(self, X, y):
        isCat=False
        if ((y==0) | (y==1)).all(): #check if the y is categorical, if yes then we save epoch accuracy
            isCat=True
            
        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.1)
            
        for epoch in range(self.epochs):
            if self.batch_size > 0: #minibatch, if batchsize = 1 then in-line trainig
                for i in range(0, len(X), self.batch_size):
                    X_batch = X[i:i + self.batch_size]
                    y_batch = y[i:i + self.batch_size]
                    self.SGD(X_batch,y_batch)
            else: #all training data for each epoch
                self.SGD(X,y)
                
            try: #if the SGD do not converge (ex. lr too big) the error would be so big that mse is nan so we catch the exception
                train_mse = mean_squared_error(self.forward(X),y)
                
                #if we use early stopping we calculate the mse even on the validation set
                if self.early_stopping:
                    val_mse = mean_squared_error(self.forward(X_val),y_val)   
                else: val_mse = 0
            except Exception as e:
                train_mse = np.nan
                val_mse = np.nan
                
            if isCat: #if y is categorical, we save the accuracy of each epoch
                pred = np.where(self.predict(X) > 0.5, 1, 0)#we get the array of classification output with a threshold of 0.5
                self.accuracy["train"].append(accuracy_score(y, pred))
                if self.early_stopping:
                    pred =  np.where(self.predict(X_val) > 0.5, 1, 0)
                    self.accuracy["validation"].append(accuracy_score(y_val, pred))
                
            self.losses["train"].append(train_mse)
            self.losses["validation"].append(val_mse)
            
            #print the epoch train and validation MSE
            if(self.verbose):
                epoch_str = f"Epoch: {epoch}"
                train_mse_str = f"Train MSE: {train_mse:.3f}"
                
                if self.early_stopping:
                    val_mse_str = f"Val MSE: {val_mse:.3f}"
                else:
                    val_mse_str=f""
                    
                mse_str = f"{train_mse_str}\t{val_mse_str}"
                separator = "=" * 40
                print(f"{separator}\n{epoch_str}\n{mse_str}")
                
            #early stopping with patience
            if self.early_stopping and epoch>1 and self.losses["validation"][-2] < self.losses["validation"][-1]:
                patience+=1
                if patience == self.patience:
                    if(self.verbose):
                        print("STOPPED FOR OVERFITTING ON VALIDATION")
                        self.t = epoch
                    break
            else:
                patience = 0
                
        self.t = self.epochs
        return
    
    def predict(self, X):
        prediction = self.forward(X)
        return prediction
        
    '''
    inizialize the weights of the network to random values, 
    it use the network structure that is provided for
    creating an array of weights
    '''
    def init_weights(self):
        layers = []
        for i in range(1, len(self.layer_structure)):
            
            if self.init_method == "rand":
                weights = np.random.rand(self.layer_structure[i-1], self.layer_structure[i]) -.1
                biases = np.zeros((1, self.layer_structure[i])) 
                
            elif self.init_method == "he":
                std_dev = np.sqrt(2 / self.layer_structure[i-1])  # He initialization std deviation
                weights = np.random.normal(0, std_dev, size=(self.layer_structure[i-1], self.layer_structure[i])) 
                biases = np.zeros((1, self.layer_structure[i]))
                
            elif self.init_method == "xavier":
                std = np.sqrt(2 / (self.layer_structure[i-1] + self.layer_structure[i]))  # Xavier initialization limit
                weights = np.random.normal(0, std, size=(self.layer_structure[i-1],self.layer_structure[i]))
                biases = np.zeros((1, self.layer_structure[i]))
                
            layers.append([weights, biases])
            
        self.netIns = [None] * len(layers)
        self.netOuts = [None] * len(layers)
        return np.array(layers, dtype=object)
    
    '''
    Calculate the forward pass of the network feeding the inputs to the nodes, i.e. multipling
    the inputs to the weights and biases and calculating the activation for each level
    '''
    def forward(self, x):
        self.netIns.clear()
        self.netOuts.clear()

        I = x  #rename vector to match typical nomenclature
        for idx, Wb in enumerate(self.weight_bias):

            self.netOuts.append(I)    #storing activations from the last layer
            I = np.dot(I, Wb[0]) + Wb[1]
            self.netIns.append(I)       #storing the inputs to the current layer
            
            #apply activation function
            if idx == len(self.weight_bias) - 1:
                out_vector = self.output_function(I)  #output layer
            else:
                I = self.hidden_function(I)  #hidden layers
        return out_vector 

    '''
    applay the SGD with different optimizers
    '''
    def SGD(self, X, y):
        weights = np.concatenate(self.weight_bias[:,0],axis=None)
        biases = np.concatenate(self.weight_bias[:,1],axis=None)

        if self.optimizer == "nesterov":
            
            if(self.momentum_schedule): #momentum slowly increasing schedule
                self.a = (1.0+np.sqrt(4.0*(self.a_prev**2)+1.0))/2
                self.momentum = (self.a_prev-1.0)/self.a
                self.a_prev=self.a

            lookahead_W = weights + self.momentum * self.last_change_W
            lookahead_b = biases + self.momentum * self.last_change_b
            self.update_Wb(lookahead_W,lookahead_b)
            lookahead_grad_w, lookahead_grad_b = self.calculate_gradient(self.forward(X),y)
    
            delta_W = (self.momentum * self.last_change_W) - (self.learning_rate * lookahead_grad_w)
            delta_b = (self.momentum * self.last_change_b) - (self.learning_rate * lookahead_grad_b)
            
        elif self.optimizer == "classic": #if momentum = 0 -> SGD
            grad_w, grad_b= self.calculate_gradient(self.forward(X),y)
            delta_W = (self.momentum * self.last_change_W) - (self.learning_rate * grad_w)
            delta_b = (self.momentum * self.last_change_b) - (self.learning_rate * grad_b)

        elif self.optimizer == "adam": 
            grad = np.concatenate(self.calculate_gradient(self.forward(X),y))
            self.t += 1
            
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2

            m_hat = self.m / (1.0 - self.beta1 ** self.t)
            v_hat = self.v / (1.0 - self.beta2 ** self.t)
                     
            delta = -(self.learning_rate * m_hat)/(np.sqrt(v_hat)+10e-8)
            delta_W, delta_b = np.split(delta, [weights.size])
            
        new_W = weights + delta_W
        new_b = biases + delta_b
        self.update_Wb(new_W,new_b)

        self.last_change_W = delta_W
        self.last_change_b = delta_b

        
    '''
    Calculates the partial derivatives of the error function with respect to the weights and biases
    it returns two vectors, one with the weights partial derivatives, from first weight of first layer,
    the other one with partial derivatives with respect of the biases, from first node of first layer
    '''    
    def calculate_gradient(self, pred, y):
        w_grad=[]
        b_grad = []
        for i in reversed(range(len(self.weight_bias))):
            if i == len(self.weight_bias)-1:   #output layer
                #derivative of error
                d_error = self.error_function(pred, y, derivative=True)
                #derivative of output function
                d_activ = self.output_function(self.netIns[i], derivative=True)
                delta = d_error * d_activ

            else: #hidden layers
                W_T = self.weight_bias[i+1][0].T
                d_error = np.dot(delta, W_T)
                d_activ = self.hidden_function(self.netIns[i],derivative=True) 
                delta = d_error *d_activ
                
            w_grad_mat = np.dot(self.netOuts[i].T , delta)
            b_grad_mat = np.sum(delta, axis=0, keepdims=True)
            
            # Add L2 regularization to weights gradient
            w_grad_mat += (self.l2_lambda * self.weight_bias[i][0])/y.shape[0] 

            # Collect gradients
            w_grad.insert(0,w_grad_mat.flatten())
            b_grad.insert(0,b_grad_mat.flatten())
        
        return np.concatenate(w_grad, axis=0),np.concatenate(b_grad, axis=0)

    '''
    this function update the weights of the network with the new weights passed
    new_W is an array of lenght |w|, the order is first value of the array is the first weight in the 
    first layer, the secon , the second of first and so on.
    '''
    def update_Wb(self, new_W, new_b):
        w_offset = 0
        b_offset = 0
        
        for i, (weight, bias) in enumerate(self.weight_bias):

            weight_size = weight.size
            bias_size = bias.size
            
            # Update weights
            self.weight_bias[i][0] = new_W[w_offset:w_offset + weight_size].reshape(weight.shape)
            w_offset += weight_size
            
            # Update biases
            self.weight_bias[i][1] = new_b[b_offset:b_offset + bias_size]
            b_offset += bias_size
    
    
    '''
    mse with l2 regularization used as loss function
    '''
    def error_function(self, pred, y, derivative=False):
        if derivative:
            error = (1/y.shape[0])*(pred - y) 
        else:
            error = (2/y.shape[0])*sum((y-pred)**2) 
            
            l2_reg = self.l2_lambda * sum(np.sum(w[0]**2) for w in self.weight_bias)
            error += l2_reg / (2 * y.shape[0])
        return error


    def reset(self):
        self.weight_bias = self.init_weights()
        self.losses = {"train": [], "validation": []}
        self.accuracy = {"train": [], "validation": []}
        self.t = 0
        self.a = 0.0
        self.a_prev = 1.0

    def set_Wb(self,Wb):
        self.weight_bias = Wb

    
    '''
    Collection of activation function that can be used
    '''
    def __sigmoid(self,x, derivative=False):
        if derivative:
            return self.__sigmoid(x) * (1.0 - self.__sigmoid(x))
        return self.__softplus(x,derivative=True)
        
    def __relu(self,x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)  #returns 1 for any x > 0, and 0 otherwise
        return np.maximum(0, x)

    def __tanh(self,x, derivative=False):
        if derivative:
            tanh_not_derivative = self.__tanh(x)
            return 1.0 - tanh_not_derivative**2
        else:
            return np.tanh(x)

    def __linear(self,x, derivative=False):
        if derivative:
            return np.ones_like(x)
        else:
            return x

    def __softplus(self, x, derivative=False):
        if derivative:
            pos_mask = (x >= 0)
            neg_mask = ~pos_mask
            z = np.zeros_like(x)
            z[pos_mask] = np.exp(-x[pos_mask])
            z[neg_mask] = np.exp(x[neg_mask])
            top = np.ones_like(x)
            top[neg_mask] = z[neg_mask]
            return top / (1. + z)
        else:
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)



