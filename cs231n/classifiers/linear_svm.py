import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # We express loss function in term of W_j and W_yi
        # So when calculate gradient we must calculate partial derivative with respect to W_j and W_yi (sub-gradient)
        
        # Collect gradient over all classes j and X training samples, hence use the += sign.
        dW[:, j] += X[i] 
        # Collect gradient over all satified classes of X[i] and all X, hence use the -= sign. 
        dW[:, y[i]] -= X[i]      


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  #
  # For mathematical convenient we multiply loss by 0.5 (convenitent when we regularize)
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization to the gradient.
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  Scores = X.dot(W)

  # Pick y_i of each X_i
  y_i = Scores[range(0, num_train), y]
  y_i = y_i.reshape((num_train, 1))

  # Calculate margin from scores
  Margin = Scores - y_i + 1

  # Solution 1: Pick loss indices that have margin greater than zero; max(0,-)  
  # These will included loss at y_i.
  # loss_idxs = np.where(M > 0)
  # Subtract the loss at y_i for each x_i
  # loss = np.sum(M[loss_idxs]) - 1*num_train 

  # Solution 2: Use numpy.maximum()
  # Remove y_i elements.
  Margin[range(0, num_train), y] = 0
  # Calculate loss from margin with max(0, -), elementwise compare.
  Loss = np.maximum(0, Margin)

  # Calculate total loss
  loss = np.sum(Loss)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # We have to collect each sub gradient of each sample (x1, x2,...)
  # First find the number of loss for each sub-W (count number of loss, W_j for j != y_i)
  
  # loss_idxs = np.where(Loss > 0)
  # Loss[loss_idxs] = 1
  Loss_count = Loss
  Loss_count[Loss > 0] = 1

  # To find the number of loss for each element W at y_i, we have to add up all loss number for each sample (row).
  # And assign it to element y_i (negative value)
  loss_row_count = np.sum(Loss_count, axis=1)
  Loss_count[range(0, num_train), y] = -loss_row_count
  
  # Then to collect sub-gradients, we multiply (dot product) this to X, 
  # But we have to transpose it first to align correctly, and then we transpose back to dW.
  dW = X.T.dot(Loss_count)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
