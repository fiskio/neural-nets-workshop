require 'nn'

local vocabSize = 5
local encodingSize = 2
local context = 2
local hiddenSize = 3

local model = nn.Sequential()
local input = nn.LookupTable(vocabSize, encodingSize)
model:add(input)
model:add(nn.Reshape(encodingSize * context))
model:add(nn.Linear(encodingSize * context, hiddenSize))
model:add(nn.Tanh())
model:add(nn.Linear(hiddenSize, encodingSize))
model:add(nn.Tanh())
local output = nn.Linear(encodingSize, vocabSize)
--output.bias:copy(unigrams:log())
model:add(output)
--model:add(nn.LogSoftMax())

local out = model:forward(torch.Tensor{1,3})
print(out)
