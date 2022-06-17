"""

	Parker Dunn
	9 Jun 2022

	Copying notes from Keras tutorial from Coursera -> Introduction to Deep Learning (CU Boulder)

"""

# A FUNCTION TO CREATE A SEQUENTIAL MODEL WITH tf.keras

def mymodel(nLayers, nNeurons, input_dim, output_dim):
	model = Sequential()
	model.add(Dense(units = nNeurons[0], activation='relu', input_dim=input_dim))
	for l in range(1,nLayers):
		model.add(Dense(units=nNeurons[l], activation='relu'))
	model.add(Dense(units=output_dim, activation='softmax'))  # softmax because we are pretending to do multiclass classification
	
	return model

## Creating a Model

model = mymodel(5, [4, 4, 4, 4, 4], 100, 5)  # 5 Layers | 4 Neurons in each layer | 100 features for input | 5 output values


## Optimizers
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])