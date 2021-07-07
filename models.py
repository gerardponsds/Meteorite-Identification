import torch.nn as nn

# CREATE THE MODEL

''' We definal a helper function to help us with the architecture design'''
def final_size(input_size,num_conv,pad=2):
  for i in range(num_conv):
    input_size = (input_size-pad)//2
  return input_size



def model(num_conv = 3,input_size = 250):

	size = final_size(input_size,num_conv)

	m = nn.Sequential(nn.Conv2d(3,64,kernel_size=3),nn.MaxPool2d(kernel_size=2),nn.ReLU(),
						  nn.Conv2d(64,64,kernel_size=3),nn.MaxPool2d(kernel_size=2),nn.ReLU(),
						  nn.Conv2d(64,128,kernel_size=3),nn.MaxPool2d(kernel_size=2),nn.ReLU(),
						  nn.Flatten(),nn.Dropout(0.1),nn.Linear(128*size*size,160),nn.ReLU(),
						  nn.Linear(160,1),nn.Sigmoid()
						  )
	return m