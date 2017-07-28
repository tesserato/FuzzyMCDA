import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

starttime = timer()

# f(x) = (a + Tanh(bx)) / 2
a = 2
b = 2

def f(x):
    return (a/2 + np.tanh(b * (x * 2 - 1))) / a # = Sig(x) / a = 1 ; b = 1/2 :: 0.1 < b < 10

def df_dx(x):
    return 1/a * (1 - ((np.tanh(b * (x * 2 - 1)) **2) )) * b * 2


class Final_Layer:
    id = 1

    def __init__(self, inputnodes):
        self.w = np.asmatrix(np.absolute(np.random.normal(0, .1, (inputnodes, 1))))
        print("Final Layer Created. id = ", Final_Layer.id, "Input Nodes = ", inputnodes)
        Final_Layer.id += 1

    def forward(self, inputs):  # type(inputs) = NORMAL Python list
        self.inputs = np.matrix(inputs)
        dot_product = np.dot(self.inputs, self.w)
        outputs = f(dot_product)
        self.partial_grad = np.multiply(inputs, df_dx(dot_product))
        return outputs.tolist()[0]  # type(returned outputs) = NORMAL Python list

    def backward(self, error): #!OK!
        self.w += self.partial_grad.T * error[0] * lr2
        berrors = (error * self.w.T / np.sum(self.w)).tolist()[0]
        return berrors # backpropagated errors !OK!
    # lr2 = 1
    # F = Final_Layer(3)
    # for _ in range(100):
    #     forw = F.forward([.3, .4, .1])
    #     federror = [.1 - forw[0]]
    #     backerror = F.backward(federror)
    #     print(federror, "|", backerror)
    #
    # print(F.forward([.3, .4, .1]))


class Binary_Layer:
    id = 1

    def __init__(self):
        self.w = np.asmatrix(np.absolute(np.random.normal(0, .1, (2, 1))))
        self.w1 = self.w[0].tolist()[0][0]
        self.w2 = self.w[1].tolist()[0][0]
        print("Binary Layer Created. id = ", Binary_Layer.id)
        Binary_Layer.id += 1

    def forward(self, inputs):  # type(inputs) = NORMAL Python list
        self.a = self.w1 + inputs[0]
        self.b = self.w2 + inputs[1]
        return [f(self.a) * f(self.b)]

    def backward(self, error):
        self.w1 += error * f(self.b) * df_dx(self.a) * lr1
        self.w2 += error * f(self.a) * df_dx(self.b) * lr1
        return error
        # if self.w1 < 0:
        #     print(self.w1 , "w1 negativo")
        #     self.w1 = 0
        # if self.w2 < 0:
        #     print(self.w2 , "w2 negativo")
        #     self.w2 = 0
# lr1 = .3
# ##helper to test binary layer independently
# bi = Binary_Layer()
# for _ in range(100):
#     forw = bi.forward([.7, .6])
#     back = bi.backward(.9 - forw[0])
#     print(forw, back)
#     forw = bi.forward([.2, .4])
#     back = bi.backward(.1 - forw[0])
#     print(forw, back)


class Network:
    def __init__(self, inputnodes):
        self.binarylayers = []
        self.results = [0 for _ in np.arange(inputnodes / 2)]
        for _ in np.arange(inputnodes / 2):
            self.binarylayers.append(Binary_Layer())
        self.finallayer = Final_Layer(len(self.binarylayers))

    def train(self, i, t):
        for n in np.arange(len(self.binarylayers)):
            self.results[n] = self.binarylayers[n].forward([i[n * 2], i[n * 2 + 1]])[0]
        o = self.finallayer.forward(self.results)
        errors = self.finallayer.backward([t[0] - o[0]])
        for m in np.arange(len(self.binarylayers)):
            self.binarylayers[m].backward(errors[m])
        return o

    def trainbin(self, i, t):
        self.finallayer.w = np.asmatrix(np.ones((len(self.binarylayers), 1)))
        for n in np.arange(len(self.binarylayers)):
            self.results[n] = self.binarylayers[n].forward([i[n * 2], i[n * 2 + 1]])[0]
        o = self.finallayer.forward(self.results)
        errors = self.finallayer.backward([t[0] - o[0]])
        self.finallayer.w = np.asmatrix(np.ones((len(self.binarylayers), 1)))
        errors = [a for a in reversed(errors)]
        for m in np.arange(len(self.binarylayers)):
            self.binarylayers[m].backward(errors[m])
        return o

    def initfinw(self):
        self.finallayer.w = np.asmatrix(np.absolute(np.random.normal(0, .1, (len(self.binarylayers), 1))))

    def trainfin(self, i, t):
        for n in np.arange(len(self.binarylayers)):
            self.results[n] = self.binarylayers[n].forward([i[n * 2], i[n * 2 + 1]])[0]

        o = self.finallayer.forward(self.results)
        self.finallayer.backward([t[0] - o[0]])
        return o

    def query(self, i):
        for n in np.arange(len(self.binarylayers)):
            self.results[n] = self.binarylayers[n].forward([i[n * 2], i[n * 2 + 1]])[0]
        return self.finallayer.forward(self.results)

    def save(self):
        weights = []
        weights.append(self.finallayer.w.shape[0])
        for b in self.binarylayers:
            weights.append(b.w1)
            weights.append(b.w2)
        for f in self.finallayer.w:
            weights.append(f.tolist()[0][0])
        return weights

    def load(self, weights): #normal list of weights
        for i in range(len(self.binarylayers)):
            self.binarylayers[i].w1 = weights[1 + 2 * i]
            self.binarylayers[i].w2 = weights[2 + 2 * i]
        self.finallayer.w = np.asmatrix(weights[weights[0] * 2 + 1:]).T
        return


def readcsv(name):
    F = open(name, 'r')
    data = []
    for line in F:
        data.append(line.replace(',', '.').split(';'))
    F.close()
    data.pop(0)
    data = np.matrix(data, dtype='float32')
    print(name, "Imported : Shape = ", data.shape, "Type = ", type(data))
    #print(data)
    return data


def normalize(matrix):
    matrix = matrix.T.tolist()
    nmatrix = []
    nrow = []
    for row in matrix:
        for cell in row:
            nrow.append( .1 + .8 * ((cell - min(row)) / (max(row) - min(row))))
        nmatrix.append(nrow)
        nrow = []
    nmatrix = np.matrix(nmatrix)
    return nmatrix.T


def denormalize(normalizedmatrix, matrix):
    normalizedmatrix = normalizedmatrix.T.tolist()
    matrix = matrix.T.tolist()
    dmatrix = []
    drow = []
    for nrow, row in zip(normalizedmatrix, matrix):
        for cell in nrow:
            drow.append(  (cell - .1) * (max(row) - min(row)) / .8 + min(row)  )
        dmatrix.append(drow)
        drow = []
    dmatrix = np.matrix(dmatrix)
    return dmatrix.T


def combine(matrix):
    intermetidate = matrix[:,:-1].T.tolist()
    final = []
    for a in itertools.combinations(intermetidate, 2):
        final.append(a[0])
        final.append(a[1])
    final = np.hstack([np.matrix(final).T,matrix[:,-1]])
    print("Data Combined : Shape = ", final.shape, "Type = ", type(final))
    #print(final)
    return final


def savecsv(file, name):
    F = open(name, 'w')
    for l in file:
        F.write(str(l).replace('[', '').replace(']', '') + '\n')
    F.close()
    print(name, "saved to disk")
    return


def trainntest(dataadress):
    rawdata = readcsv(dataadress)
    traindata = normalize(rawdata)
    traindata = combine(traindata)
    network = Network(traindata.shape[1] - 1)
    #print(traindata.shape[1] - 1)
    errorseries = []
    queryerrors = []
    periodicerror = []
    results = []
    targets = []
    lessons = 8000
    reportevery = 500
    print("Training binary Layers : \n")
    for counter in np.arange(0):
        traindata = traindata.tolist()
        random.shuffle(traindata)
        traindata = np.matrix(traindata)
        for d in traindata:
            i = d.tolist()[0][:-1]
            t = [d.tolist()[0][-1]]
            #print(i)
            periodicerror.append(np.abs(np.array(t) - np.array(network.trainbin(i, t))))

        if counter % reportevery == 0:
            errorseries.append([str(counter) + ' , ' + str(np.mean(periodicerror)) + ' , ' + str(network.save())])
            print(errorseries[-1][0])
            periodicerror = []

    print("\nTraining Final Layer : \n")
    network.initfinw()
    for counter in np.arange(0):
        traindata = traindata.tolist()
        random.shuffle(traindata)
        traindata = np.matrix(traindata)
        for d in traindata:
            i = d.tolist()[0][:-1]
            t = [d.tolist()[0][-1]]
            #print(i)
            periodicerror.append(np.abs(np.array(t) - np.array(network.trainfin(i, t))))

        if counter % reportevery == 0:
            errorseries.append([str(counter) + ' , ' + str(np.mean(periodicerror)) + ' , ' + str(network.save())])
            print(errorseries[-1][0])
            periodicerror = []

    print("\nTraining Both Layers : \n")
    for counter in np.arange(lessons+1):
        traindata = traindata.tolist()
        random.shuffle(traindata)
        traindata = np.matrix(traindata)
        for d in traindata:
            i = d.tolist()[0][:-1]
            t = [d.tolist()[0][-1]]
            #print(i)
            periodicerror.append(np.abs(np.array(t) - np.array(network.train(i, t))))

        if counter % reportevery == 0:
            errorseries.append([str(counter) + ' , ' + str(np.mean(periodicerror)) + ' , ' + str(network.save())])
            print(errorseries[-1][0])
            periodicerror = []


    for d in traindata:
        i = d.tolist()[0][:-1]
        t = d.tolist()[0][-1:]
        o = network.query(i)
        queryerrors.append(np.abs(np.array(t) - np.array(o)))
        print('Tgt =', t, '; Out =', o)
        targets.append(t)
        results.append(o)
    print('Querying error = ', np.mean(queryerrors))
    endtime = timer()
    print('Execution Time =', endtime - starttime, 'seconds')
    print("TestQuery with I = .5 ... .5: ", network.query([.5 for _ in range(traindata.shape[1] - 1)]))
    targets = denormalize(np.matrix(targets)[:,-1], rawdata[:,-1])
    results = denormalize(np.matrix(results)[:,-1], rawdata[:,-1])
    tgtnres = np.hstack([targets, results])

    return tgtnres, errorseries, network.save()

lr1 = 0.0005 #.001
lr2 = 0.3 #.3

#results, series, weights = trainntest("data.txt")

rawdata = readcsv("qdata.txt")
traindata = normalize(rawdata)
traindata = combine(traindata)

testnet = Network(traindata.shape[1] - 1)
testnet.load([15, 0.081829906429125368, 0.062344003188603038, -0.029306536416474207, 0.27209927806181466, 0.070313992160749308, 0.10452553146127451, -0.0027652857194215273, 0.059614637815118492, 0.027972270911631434, 0.034274888091544164, -0.063361449215695531, 0.28558882783435607, 0.081497838854157761, 0.13726556876939239, 0.073401061775850526, 0.18158862827119598, 0.0688510068137764, 0.12991479757042493, 0.10129477201038647, 0.094010958764267974, 0.10890167364226927, -0.03352137588675335, 0.11612916944717887, 0.032298922583439048, 0.052459760227330308, 0.028543955009082904, 0.044146554696434151, 0.06163234599126937, 0.39608629992745786, 0.094486010523097949, 0.5889766760674534, 2.9060470346344065, -1.1058400254632925, 0.46923985462318757, -2.3595849329682674, 2.244948618858462, -1.085756420732917, 0.2764093273057685, -1.8854002461115333, 0.1637172696661886, 0.08636569946143469, -1.2563976236704768, -0.03759986658625611, -1.3217388446324516, 2.823273158704356])
print("TestQuery with I = .5 ... .5: ", testnet.query([.5 for _ in range(traindata.shape[1] - 1)]))

# print(weights)
# savecsv(series, "00 evow.txt")
# savecsv(weights, "01 finalw.txt")
# savecsv(results, "02 results.txt")
#
# print(weights)

res = []
for q in traindata:
    res.append(testnet.query(q.tolist()[0][:-1]))

results = denormalize(np.matrix(res)[:,-1], rawdata[:,-1])

for l in results:
    print(l)