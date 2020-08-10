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
onehot <- function(Y){
  num_class <- length(unique(Y))
  oh <- matrix(0,nrow=length(Y),ncol = num_class)
  for(i in 1:length(Y)){
    oh[i,Y[i]] <- 1
  }
  oh
}


X <- as.matrix(iris[,-5])
Y <- onehot(as.numeric(iris[,5]))

inputdim <- 4
layers <- c(2)
outputdim <- 3
# TODO activation : sigmoid(Done), relu, tanh, elu
# TODO output_activation : softmax(Done), sigmoid, linear
# TODO loss: CE(Done), MSE
# TODO L1, L2 regularization
# TODO SGD, MOMENTUM, ADAM, RMSPROP

# predict
predict_JSNN <- function(x){
  os <- list()
  zs <- list()
  # o <- x%*%weights[[1]]+bias[[1]]
  o <- sweep(x%*%weights[[1]],2,bias[[1]],`+`)
  os[[1]] <- o
  z <- sigmoid(o)
  zs[[1]] <- z
  
  if(length(layers)>1){
    for(i in 2:(length(layers))){
      # o <- z%*%weights[[i]]+bias[[i]]
      o <- sweep(z%*%weights[[i]],2,bias[[i]],`+`)
      os[[i]] <- o
      z <- sigmoid(o)
      zs[[i]] <- z
    }
  }

  
  # o <- z%*%weights[[length(layer_structure)-1]]+bias[[length(layer_structure)-1]]
  o <- sweep(z%*%weights[[length(layer_structure)-1]],2,bias[[length(layer_structure)-1]],`+`)
  os[[length(layer_structure)-1]] <- o
  
  z <- t(apply(o,1,function(x){softmax(x)}))
  zs[[length(layer_structure)-1]] <- z
  return(list(output=z,os=os,zs=zs))
}

# LOSS
multiclass_cross_entropy <- function(real, pred){
  -sum(real*log(pred))/nrow(pred)
}

# gradients
gradients <- function(X,Y){
  t_output <- predict_JSNN(X)
  pred <- t_output$output
  real <- Y
  N <- nrow(pred)
  
  grad_weights <- list()
  grad_bias <- list()
  
  # output gradient
  dZ <- (pred-real)
  dW <- t((t(dZ) %*% (t_output$zs[[length(weights)-1]]))/N)
  db <- colSums(dZ)/N
  dA_back <- t(weights[[length(weights)]] %*% t(dZ))
  
  grad_weights[[length(weights)]] <- dW
  grad_bias[[length(bias)]] <- db
  # internal gradient
  for(i in (length(weights)-1):1){
    dZ <- dA_back*sigmoid_derivative(t_output$os[[i]])
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
# grads <- gradients(X,Y)


# initailize
weights <- list()
bias <- list()
layer_structure <- c(inputdim, layers, outputdim)

for(i in 1:(length(layer_structure)-1)){
  weights[[i]] <- matrix(rnorm(prod(layer_structure[c(i,i+1)]),mean = 0,sd = 0.1),
                         layer_structure[i],
                         layer_structure[i+1])
  bias[[i]] <- rnorm(layer_structure[i+1],mean = 0,sd = 0.1)
}


# fit
lr <- 0.1

ces <- c()
for(epoch in 1:10000){
  grads <- gradients(X,Y)
  for(i in 1:length(weights)){
    weights[[i]] <- weights[[i]]-lr*grads$grad_weights[[i]]
    bias[[i]] <- bias[[i]]-lr*grads$grad_bias[[i]]
  }
  ces <- c(ces,multiclass_cross_entropy(real = Y,pred = predict_JSNN(X)$output))
  
}
plot(ces)


# measure
real = Y
pred = predict_JSNN(X)$output
table(apply(Y,1,which.max),apply(pred,1,which.max))


################
# library(numDeriv)
# gradients(X,Y)
# 
# weights
# dims <- sapply(weights,dim)
# 
# weights_vec <- unlist(weights)
# 
# relist_weight <- function(weights_vec,dims){
#   weights <- list()
#   gidx <- 1
#   for(i in 1:ncol(dims)){
#     size <- prod(dims[,i])
#     weights[[i]] <- matrix(weights_vec[gidx:(gidx+size-1)],dims[1,i],dims[2,i])
#     gidx <- (gidx+size)
#   }
#   weights
# }
# 
# 
# 
# 
# func1 <- function(weights_vec){
#   # X,Y
#   # weights_vec
#   x <- X
#   predict_JSNN_weights_input <- function(weights_vec){
#     
#     weights <- relist_weight(weights_vec,dims)
#     os <- list()
#     zs <- list()
#     
#     o <- x%*%weights[[1]]+bias[[1]]
#     os[[1]] <- o
#     z <- sigmoid(o)
#     zs[[1]] <- z
#     
#     for(i in 2:(length(layers))){
#       o <- z%*%weights[[i]]+bias[[i]]
#       os[[i]] <- o
#       z <- sigmoid(o)
#       zs[[i]] <- z
#     }
#     
#     o <- z%*%weights[[length(layer_structure)-1]]+bias[[length(layer_structure)-1]]
#     os[[length(layer_structure)-1]] <- o
#     
#     z <- t(apply(o,1,function(x){softmax(x)}))
#     zs[[length(layer_structure)-1]] <- z
#     return(list(output=z,os=os,zs=zs))
#   }
#   
#   t <- predict_JSNN_weights_input(weights_vec)
#   cross_entropy(real = Y,pred = t$output)
# }
# 
# grad_numeric <- grad(func1,weights_vec, method="simple")
# 
# grad_exact <- gradients(X,Y)
# unlist(grad_exact$grad_weights)
# 
# plot((grad_numeric-unlist(grad_exact$grad_weights)))