import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

###################################################################
# Define Helpers                                                  #
###################################################################
def readCsv(path):
  readings = []
  with open(path) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
      readings.append(Reading(row[0], row[1]))
  # Add ZScores
  yValues = list(map(lambda r: r.y, readings))
  zScores = np.abs(stats.zscore(yValues))
  for i,r in enumerate(readings):
    r.z = zScores[i]
  return readings

class Reading(object):
  def __getitem__(self, key):
    return getattr(self, str(key))
  def __setitem__(self, key, value):
    return setattr(self, str(key), value)
  def __str__(self):
    return 'R' + str(self.__dict__)
  def __init__(self, t, y):
    self.t = t
    self.y = y

def mapToQuad(xList, quadParams):
  def quad(x, a, b, c):
    return (a * (x ** 2)) + (b * x) + c
  return list(map(lambda x: quad(x, *quadParams), xList))
  # return np.add(np.multiply(a, np.power(x, 2)), np.multiply(b, x), c)

def _map(values, attribute):
  return list(map(lambda r: r[attribute], values))

def filterByIndexList(dataList, indexList): 
    return [dataList[i] for i in indexList]

def correctReadings(p0, p, yList):
  t = np.linspace(0, 10, num=100)
  ideal = mapToQuad(t, p0)
  realized = mapToQuad(t, p)
  correctionParams = np.polyfit(realized, ideal, 2)
  return mapToQuad(yList, correctionParams)

###################################################################
# Processing Starts Here                                          #
###################################################################
r0 = readCsv('./sensor_0.csv')
r1 = readCsv('./sensor_1.csv')
r2 = readCsv('./sensor_2.csv')

# Define Figure
fig, axs = plt.subplots(2, 3, figsize=(12, 9))

# Plot 1. Original Readings
axs[0][0].scatter(_map(r0, 't'), _map(r0, 'y'), marker='.', alpha=0.2, label='s0')
axs[0][0].scatter(_map(r1, 't'), _map(r1, 'y'), marker='.', alpha=0.2, label='s1')
axs[0][0].scatter(_map(r2, 't'), _map(r2, 'y'), marker='.', alpha=0.2, label='s2')
axs[0][0].set_title('1. Original Readings')

# Remove Outliers
inlierMaxZScore = 0.6 # about half a std dev
inlierIndicies = list(filter(lambda i: (abs(r0[i]['z']) < inlierMaxZScore and abs(r1[i]['z']) < inlierMaxZScore and abs(r2[i]['z']) < inlierMaxZScore), range(len(r0))))
r0 = filterByIndexList(r0, inlierIndicies)
r1 = filterByIndexList(r1, inlierIndicies)
r2 = filterByIndexList(r2, inlierIndicies)

# Plot 2. Inlier Readings
axs[0][1].scatter(_map(r0, 't'), _map(r0, 'y'), marker='.', alpha=0.2, label='s0')
axs[0][1].scatter(_map(r1, 't'), _map(r1, 'y'), marker='.', alpha=0.2, label='s1')
axs[0][1].scatter(_map(r2, 't'), _map(r2, 'y'), marker='.', alpha=0.2, label='s2')
axs[0][1].set_title('2. Inlier Readings')

# Fit Curves
p0 = np.polyfit(_map(r0, 't'), _map(r0, 'y'), 2)
p1 = np.polyfit(_map(r1, 't'), _map(r1, 'y'), 2)
p2 = np.polyfit(_map(r2, 't'), _map(r2, 'y'), 2)

# Plot 3. Fit Curves
axs[0][2].scatter(_map(r0, 't'), mapToQuad(_map(r0, 't'), p0), marker='.', alpha=0.2, label='s0'+str(p0))
axs[0][2].scatter(_map(r1, 't'), mapToQuad(_map(r1, 't'), p1), marker='.', alpha=0.2, label='s1'+str(p1))
axs[0][2].scatter(_map(r2, 't'), mapToQuad(_map(r2, 't'), p2), marker='.', alpha=0.2, label='s2'+str(p2))
axs[0][2].set_title('3. Fit Curves (ax^2 + bx + c)')

# Plot 4. Corrected Readings
y1Corrected = correctReadings(p0, p1, _map(r1, 'y'))
y2Corrected = correctReadings(p0, p2, _map(r2, 'y'))
axs[1][0].scatter(_map(r0, 't'), _map(r0, 'y'), marker='.', alpha=0.2, label='s0')
axs[1][0].scatter(_map(r1, 't'), y1Corrected, marker='.', alpha=0.2, label='s1')
axs[1][0].scatter(_map(r2, 't'), y2Corrected, marker='.', alpha=0.2, label='s2')
axs[1][0].set_title('4. Corrected Readings')

# Plot 5. Correlation
axs[1][1].scatter(_map(r0, 'y'), _map(r0, 'y'), marker='.', alpha=0.2, label='s0')
axs[1][1].scatter(_map(r0, 'y'), y1Corrected, marker='.', alpha=0.2, label='s1')
axs[1][1].scatter(_map(r0, 'y'), y2Corrected, marker='.', alpha=0.2, label='s2')
axs[1][1].set_title('5. Correlation')

# Plot 6. Residual
axs[1][2].scatter(_map(r1, 't'), np.subtract(y1Corrected, _map(r0, 'y')), marker='.', alpha=0.2, label='s1')
axs[1][2].scatter(_map(r2, 't'), np.subtract(y2Corrected, _map(r0, 'y')), marker='.', alpha=0.2, label='s2')
axs[1][2].set_title('6. Residual (corrected - reference)')

# Show Plots
for ax in axs.reshape(-1):
  ax.legend()
  ax.grid(True)
plt.tight_layout()
plt.show()