import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)

  for i in range(num_train):
    # Normalize to prevent numerical unstability.
    score_i = scores[i]
    normalized_score_i = score_i - np.amax(score_i)
    sum_e_score_i = np.sum(np.e**normalized_score_i)
    loss += - np.log( np.e**normalized_score_i[y[i]] / sum_e_score_i )
    
    for j in range(num_class):
      if j != y[i]:
        dW[:, j] += (np.e**normalized_score_i[j] / sum_e_score_i) * X[i]
     
    dW[:, y[i]] += ((np.e**normalized_score_i[y[i]] - sum_e_score_i) / sum_e_score_i) * X[i]
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  # Add regularization to the gradient.
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]  
  scores = X.dot(W)
  
  # Normalize to prevent numerical unstability.
  n_score = scores - np.amax(scores, axis=1)[:, np.newaxis]
  
  e_n_score = np.e**n_score
  sum_e_n_score = np.sum(e_n_score, axis=1)[:, np.newaxis]
  k = e_n_score / sum_e_n_score
  k_y_i = k[range(0, num_train), y]

  loss = np.sum(-np.log(k_y_i))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  k[range(0, num_train), y] = k_y_i - 1
  dW = X.T.dot(k)
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

