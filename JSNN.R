# functions
sigmoid <- function(x){
  1/(1+exp(-x)) 
}
sigmoid_derivative <- function(Z){
  s = sigmoid(Z)
  s * (1 - s)
}
softmax <- function(x){
  z <- x-max(x)
  exp(z)/sum(exp(z))
}
softmax_caterogical_cross_entropy_derivative <- function(pred,real){
  pred-real
}
sigmoid_binary_cross_entropy_derivative <- function(pred,real){
  pred-real
}

# LOSS
caterogical_cross_entropy <- function(real, pred){
  -sum(real*log(pred))/nrow(pred)
}
binary_cross_entropy <- function(real, pred){
  -sum(real*log(pred)+(1-real)*log(1-pred))/length(real)
}

onehot <- function(Y){
  num_class <- length(unique(Y))
  oh <- matrix(0,nrow=length(Y),ncol = num_class)
  for(i in 1:length(Y)){
    oh[i,Y[i]] <- 1
  }
  oh
}

#################################################################################

# TODO activation : sigmoid(Done), relu, tanh, elu
# TODO output_activation : softmax(Done), sigmoid, linear
# TODO loss: CE(Done), MSE
# TODO L1, L2 regularization
# TODO SGD, MOMENTUM, ADAM, RMSPROP


# initailize
initialize_weights <- function(inputdim, layers, outputdim){
  weights <- list()
  bias <- list()
  layer_structure <- c(inputdim, layers, outputdim)
  
  for(i in 1:(length(layer_structure)-1)){
    weights[[i]] <- matrix(rnorm(prod(layer_structure[c(i,i+1)]),mean = 0,sd = 0.1),
                           layer_structure[i],
                           layer_structure[i+1])
    bias[[i]] <- rnorm(layer_structure[i+1],mean = 0,sd = 0.1)
  }
  return(list(weights=weights,bias=bias))
}
# weights_bias <- initialize_weights(inputdim, layers, outputdim)

# model
model <- function(inputdim,layers,outputdim,activations,output_activation,loss_function){
  if(length(layers)!=length(activations)){
    stop('length of layers and activations must be SAME length.')
  }
  weights_bias <- initialize_weights(inputdim, layers, outputdim)
  layer_structure <- c(inputdim, layers, outputdim)
  
  output <- list()
  output$inputdim <- inputdim
  output$layers <- layers
  output$outputdim <- outputdim
  output$layer_structure <- layer_structure
  output$activations <- activations
  output$output_activation <- output_activation
  output$loss_function <- loss_function
  
  output$weights <- weights_bias$weights
  output$bias <- weights_bias$bias
  
  return(output)
}


# predict
predict_JSNN <- function(model,x){
  os <- list()
  zs <- list()
  # o <- x%*%weights[[1]]+bias[[1]]
  o <- sweep(x%*%model$weights[[1]],2,model$bias[[1]],`+`)
  os[[1]] <- o
  z <- get(model$activations[1])(o)
  zs[[1]] <- z
  
  if(length(model$layers)>1){
    for(i in 2:(length(model$layers))){
      # o <- z%*%weights[[i]]+bias[[i]]
      o <- sweep(z%*%model$weights[[i]],2,model$bias[[i]],`+`)
      os[[i]] <- o
      z <- get(model$activations[i])(o)
      zs[[i]] <- z
    }
  }

  
  # o <- z%*%weights[[length(layer_structure)-1]]+bias[[length(layer_structure)-1]]
  o <- sweep(z%*%model$weights[[length(model$layer_structure)-1]],2,model$bias[[length(model$layer_structure)-1]],`+`)
  os[[length(model$layer_structure)-1]] <- o
  
  z <- t(apply(o,1,function(x){get(model$output_activation)(x)}))
  zs[[length(model$layer_structure)-1]] <- z
  return(list(output=z,os=os,zs=zs))
}
# predict_JSNN(JSNN_model,X)


# gradients
gradients <- function(model,X,Y,weights,bias){
  t_output <- predict_JSNN(model,X)
  pred <- t_output$output
  real <- Y
  N <- nrow(pred)
  
  grad_weights <- list()
  grad_bias <- list()
  
  # output gradient
  dZ <- get(paste0(model$output_activation,'_',model$loss_function,'_derivative'))(pred,real)
  dW <- t((t(dZ) %*% (t_output$zs[[length(weights)-1]]))/N)
  db <- colSums(dZ)/N
  dA_back <- t(weights[[length(weights)]] %*% t(dZ))
  
  grad_weights[[length(weights)]] <- dW
  grad_bias[[length(bias)]] <- db
  # internal gradient
  for(i in (length(weights)-1):1){
    dZ <- dA_back*get(paste0(model$activations[i],'_derivative'))(t_output$os[[i]])
    # dZ <- dA_back*sigmoid_derivative(t_output$os[[i]])
    if(i==1){
      dW <- 1./N*(t(t(dZ) %*% (X)))
    }else{
      dW <- 1./N*(t(t(dZ) %*% (t_output$zs[[i-1]])))
    }
    db <- colSums(dZ)/N
    dA_back <- t(weights[[i]] %*% t(dZ))
    
    grad_weights[[i]] <- dW
    grad_bias[[i]] <- db
  }
  
  return(list(grad_weights=grad_weights,grad_bias=grad_bias))
}
# gradients(JSNN_model,X,Y,JSNN_model$weights,JSNN_model$bias)


# fit
fit <- function(model,X,Y,learning_rate,epoch){
  lr <- learning_rate
  weights <- model$weights
  bias <- model$bias
  ces <- c()
  for(epo in 1:epoch){
    grads <- gradients(model,X,Y,weights,bias)
    for(i in 1:length(weights)){
      weights[[i]] <- weights[[i]]-lr*grads$grad_weights[[i]]
      bias[[i]] <- bias[[i]]-lr*grads$grad_bias[[i]]
    }
    model$weights <- weights
    model$bias <- bias
    loss <- caterogical_cross_entropy(real = Y,pred = predict_JSNN(model,X)$output)
    ces <- c(ces,loss)
    
    if(epo%%100==0){
      cat(sprintf("Epoch : %04d    loss = %.4f \n",epo,loss))  
    }
    
  }
  model$weights <- weights
  model$bias <- bias
  plot(ces)
  return(model)
}

#################################################################################

X <- as.matrix(iris[,-5])
Y <- onehot(as.numeric(iris[,5]))

inputdim <- 4
layers <- c(16,16)
activations <- c("sigmoid","sigmoid")
outputdim <- 3
output_activation <- "softmax"
loss_function <- "caterogical_cross_entropy"

JSNN_model <- model(inputdim,layers,outputdim,activations,output_activation,loss_function)
JSNN_model <- fit(JSNN_model,X,Y,0.1,10000)

# measure
real = Y
pred = predict_JSNN(JSNN_model,X)$output
table(apply(Y,1,which.max),apply(pred,1,which.max))
