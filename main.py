import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
plt.style.use('ggplot')


# Model
def sirModel(alpha, beta, variables):
    s = variables[0]
    i = variables[1]
# r = variables[2]
    sDot = -beta * s * i
    iDot = beta * s * i - alpha * i
    rDot = alpha * i
    return [sDot, iDot, rDot]


# example
exalpha = 0.1
exbeta = 1.2
exampleCondition = np.array([0.99, 0.01, 0.0])


def exampleModel(startingcondition, t):
    return sirModel(exalpha, exbeta, startingcondition)


# time points
time = np.linspace(0, 50, num=100)

exampleSolution = odeint(exampleModel, exampleCondition, time)


# example Data (with error)

def addnoise(correct, noise):
    return [cor + noi for cor, noi in zip(correct, noise)]


noise = [np.random.uniform(low=-0.05, high=0.05) for val in exampleSolution[:, 1]]

data = addnoise(exampleSolution[:, 1], noise)


# fitting for I

def solver(t, alpha, beta, startingcondition):
    def localModel(localstartingcondition, tt):
        return sirModel(alpha, beta, localstartingcondition)

    localsolution = odeint(localModel, startingcondition, t)

    return localsolution[:, 1]


def fitter(t, alpha, beta):
    return solver(t, alpha, beta, exampleCondition)


# loss Function (squared difference)

def lossfunction(paramguess):
    try:
        fittedparams = curve_fit(fitter, time, data, p0=paramguess)
        solution = solver(time, fittedparams[0][0], fittedparams[0][1], exampleCondition)

        loss = [(comp - dat) ** 2 for comp, dat in zip(solution, data)]
    except RuntimeError:
        loss = [17, 17]
    return sum(loss)


# Random selection of the initial guess values for alpha and beta
def RandSearch(iterationnuber):
    GuessList = []
    LossList = []

    for k in range(iterationnuber):
        localGuess = [np.random.uniform(low=0, high=2) for val in range(2)]
        localLoss = lossfunction(localGuess)

        GuessList.append(localGuess)  # Contains the list of random initial values
        LossList.append(localLoss)  # List of the Losses

    minLoss = np.min(LossList)  # Min Loss value
    minLocation = [j for j in range(iterationnuber) if
                   LossList[j] == minLoss]  # Returns the location of the min value in the list

    bestGuess = GuessList[minLocation[0]]  # Best initial value guess

    return GuessList, bestGuess


errs, bG = RandSearch(10)

fittedParams = curve_fit(fitter, time, data, bG)
fittedSolution = solver(time, fittedParams[0][0], fittedParams[0][1], exampleCondition)

plt.plot(time, fittedSolution, 'b-', linewidth=2, label='fitted plot')
plt.plot(time, data, 'g.', linewidth=2, label='data')
plt.xlabel('time')
plt.ylabel('infected')
plt.legend()
plt.show()
