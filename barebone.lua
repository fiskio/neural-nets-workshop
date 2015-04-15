require 'nn'

function cprint(str) print(sys.COLORS.green..str..) end

-- setting the random generator seed for reproducibily
torch.manualSeed(42)

local embeddingSize = 4
local contextLength = 3
local vocabSize = 10
local hiddenSize = 5
local stdv = 1
-- Vocabulary

local lookupTable = torch.FloatTensor(embeddingSize, vocabSize):uniform(-stdv, stdv)
cprint('LookupTable of '..vocabSize..' words, each is a vector of size '..embeddingSize)
print(lookupTable)

-- Hidden

local contextToHidden = torch.FloatTensor(hiddenSize, contextLength * embeddingSize):uniform(-stdv, stdv)

cprint('Context-To-Hidden matrix is:')
print(contextToHidden)

local hiddenToEmbedding = torch.FloatTensor(embeddingSize, hiddenSize):uniform(-stdv, stdv)

cprint('Hidden-To-Embedding matrix is:')
print(hiddenToEmbedding)

local embeddingToVocabulary = torch.FloatTensor(vocabSize, embeddingSize):uniform(-stdv, stdv)

cprint('Embedding-To-Vocabulary matrix is:')
print(embeddingToVocabulary)

function softMax(matrix)
   -- -log(sum(exp(matrix)) * 1/exp(matrix))
   return torch.mul(torch.exp(matrix):pow(-1), torch.sum(torch.exp(matrix), 1)[1]):log():mul(-1)
end

-- Forward
--[[
local oneHot = torch.FloatTensor():eye(vocabSize)
cprint('1-Hot representation of second word is:')
print(oneHot[2])

local secondWord = torch.mv(lookupTable, oneHot[2])
cprint('Vector representation of second word is:')
print(secondWord)
--]]
local oneHotIndices = torch.LongTensor{1,3,5}
cprint('Context will be built of words at indices...')
print(oneHotIndices)

local contextMatrix = lookupTable:index(2, oneHotIndices)
cprint('...which corresponds to the following matrix:')
print(oneHotContext)
--[[
local contextMatrix = torch.mm(lookupTable, oneHotContext)
cprint('The corresponding matrix of embeddings are:')
print(contextMatrix)
--]]
local contextVector = torch.reshape(contextMatrix, contextMatrix:nElement())
cprint('...which reshaped as a vector is:')
print(contextVector)

local output = contextToHidden * contextVector
print(output)

output = torch.mv(hiddenToEmbedding, output)
print(output)

output = torch.mv(embeddingToVocabulary, output)
print(output)

output = softMax(output)
print(output)
